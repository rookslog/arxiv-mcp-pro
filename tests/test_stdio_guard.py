"""Tests for the stdio protocol-channel guard.

The regression these cover: under the stdio transport, stdout is the JSON-RPC
channel. A library that prints to stdout injects non-JSON bytes mid-stream, the
client's decoder fails on the first stray character, and the server dies of the
resulting broken pipe. PyMuPDF does exactly this — it binds ``sys.stdout`` at
import time and prints MuPDF diagnostics through it.

The interesting assertions need real file descriptors, which pytest's capture
machinery replaces, so those run in a subprocess against a real pipe.
"""

import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arxiv_mcp_server import server as server_module
from arxiv_mcp_server.stdio_guard import protected_stdout


def _run_child(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a child process with real pipes for stdout and stderr."""
    script = textwrap.dedent("""
            import os, sys
            from arxiv_mcp_server.stdio_guard import protected_stdout
            """) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_stray_python_writes_do_not_reach_the_protocol_channel():
    """print() inside the guard goes to stderr; only protocol bytes reach stdout."""
    result = _run_child("""
        with protected_stdout() as protocol:
            print("STRAY_PRINT")
            sys.stdout.write("STRAY_WRITE\\n")
            protocol.write('{"jsonrpc":"2.0","id":1}\\n')
            protocol.flush()
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":1}\n'
    assert "STRAY_PRINT" in result.stderr
    assert "STRAY_WRITE" in result.stderr


def test_writes_through_a_pre_captured_stdout_do_not_reach_the_channel():
    """The PyMuPDF failure mode: a reference to stdout bound before the guard ran.

    A Python-level ``sys.stdout`` reassignment would not intercept this; the
    file-descriptor redirect does.
    """
    result = _run_child("""
        captured = sys.stdout          # what pymupdf does at import time
        with protected_stdout() as protocol:
            print("MuPDF error: cannot recognize xref", file=captured, flush=True)
            protocol.write('{"jsonrpc":"2.0","id":2}\\n')
            protocol.flush()
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":2}\n'
    assert "cannot recognize xref" in result.stderr


def test_raw_descriptor_writes_do_not_reach_the_channel():
    """C extensions write to fd 1 directly; MuPDF is a C library."""
    result = _run_child("""
        with protected_stdout() as protocol:
            os.write(1, b"C_LEVEL_DIAGNOSTIC\\n")
            protocol.write('{"jsonrpc":"2.0","id":3}\\n')
            protocol.flush()
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":3}\n'
    assert "C_LEVEL_DIAGNOSTIC" in result.stderr


def test_diagnostics_are_redirected_not_discarded():
    """Stray output must remain visible on stderr for operators to debug with."""
    result = _run_child("""
        with protected_stdout() as protocol:
            print("keep me visible")
            protocol.write("{}\\n")
            protocol.flush()
        """)

    assert result.returncode == 0, result.stderr
    assert "keep me visible" in result.stderr
    assert "keep me visible" not in result.stdout


def test_stdout_is_restored_when_the_session_ends():
    """An embedding host must get its stdout back; a second session must work.

    Without restoration the second guard would duplicate the already-redirected
    descriptor and answer onto stderr instead of the client.
    """
    result = _run_child("""
        with protected_stdout() as protocol:
            protocol.write("FIRST\\n"); protocol.flush()
        print("HOST_OUTPUT_AFTER_SESSION")          # must reach real stdout again
        with protected_stdout() as protocol:
            protocol.write("SECOND\\n"); protocol.flush()
            print("STRAY_IN_SECOND")                # must still be diverted
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "FIRST",
        "HOST_OUTPUT_AFTER_SESSION",
        "SECOND",
    ]
    assert "STRAY_IN_SECOND" in result.stderr


def test_fd_one_is_diverted_even_when_stderr_has_no_descriptor():
    """With stderr unusable, fd 1 must still stop being the protocol channel.

    Reassigning sys.stdout alone would leave raw os.write(1, ...) and C-level
    writes injecting into JSON-RPC — the exact failure this guard prevents.
    Diagnostics are lost in this case; keeping the protocol intact wins.
    """
    result = _run_child("""
        os.close(2)                                  # stderr has no descriptor
        with protected_stdout() as protocol:
            os.write(1, b"MUST_NOT_APPEAR\\n")
            protocol.write('{"jsonrpc":"2.0","id":4}\\n')
            protocol.flush()
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":4}\n'
    assert "MUST_NOT_APPEAR" not in result.stdout


def test_a_live_fd_two_survives_a_replaced_sys_stderr():
    """A host may swap sys.stderr for an object with no fileno while fd 2 lives.

    The guard must not treat that as proof descriptor 2 is dead: clobbering it
    and closing it on the way out would destroy a descriptor the guard does not
    own, leaving the host's stderr permanently EBADF.
    """
    result = _run_child(r"""
        import io
        sys.stderr = io.StringIO()          # fd 2 is still perfectly alive
        with protected_stdout() as protocol:
            protocol.write("PROTOCOL\n"); protocol.flush()
        os.write(2, b"HOST_STDERR_STILL_WORKS\n")
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "PROTOCOL\n"
    assert "HOST_STDERR_STILL_WORKS" in result.stderr


def test_buffered_writes_to_a_captured_stdout_never_reach_the_channel():
    """Unflushed diagnostics must not land on the protocol after restoration.

    A library holding a pre-guard stdout reference may write without flushing —
    normal when stdout is a pipe. Those bytes sit in the wrapper's buffer; if
    the guard restores fd 1 before flushing it, a later flush empties them
    straight onto the JSON-RPC channel.
    """
    result = _run_child(r"""
        captured = sys.stdout
        with protected_stdout() as protocol:
            print("BUFFERED_STRAY", file=captured)     # deliberately not flushed
            protocol.write('{"jsonrpc":"2.0","id":9}\n')
            protocol.flush()
        captured.flush()                                # after fd 1 is restored
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":9}\n'
    assert "BUFFERED_STRAY" in result.stderr


def test_degrades_safely_without_a_stdout_descriptor(monkeypatch):
    """Under capture or embedding, fall back to the Python-level redirect."""

    class _NoFileno:
        def flush(self):
            pass

        def fileno(self):
            raise OSError("no descriptor")

    sentinel_stdout = _NoFileno()
    sentinel_stderr = object()
    monkeypatch.setattr(sys, "stdout", sentinel_stdout)
    monkeypatch.setattr(sys, "stderr", sentinel_stderr)

    with protected_stdout() as returned:
        assert returned is sentinel_stdout
        assert sys.stdout is sentinel_stderr

    assert sys.stdout is sentinel_stdout  # restored on exit


@pytest.mark.asyncio
async def test_run_stdio_hands_the_protected_stream_to_the_transport():
    """_run_stdio must give the transport the protected stream, not sys.stdout."""
    protected = MagicMock(name="protocol_stream")
    wrapped = MagicMock(name="wrapped")
    guard = MagicMock()
    guard.__enter__ = MagicMock(return_value=protected)
    guard.__exit__ = MagicMock(return_value=False)
    transport = MagicMock()
    transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    transport.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(
            server_module, "protected_stdout", return_value=guard
        ) as guard_factory,
        patch.object(server_module.anyio, "wrap_file", return_value=wrapped) as wrap,
        patch.object(
            server_module, "stdio_server", return_value=transport
        ) as stdio_server,
        patch.object(server_module.server, "run", new_callable=AsyncMock) as run,
    ):
        await server_module._run_stdio()

    guard_factory.assert_called_once_with()
    guard.__exit__.assert_called_once()  # released even on the happy path
    wrap.assert_called_once_with(protected)
    stdio_server.assert_called_once_with(stdout=wrapped)
    run.assert_awaited_once()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Aliasing detection is POSIX-only. Windows anonymous pipes carry no "
        "filesystem identity — two independent pipes both report zero "
        "st_dev/st_ino — so _same_destination deliberately treats that as "
        "inconclusive rather than lose every diagnostic to the null device on "
        "the normal Windows arrangement. Closing this needs a Win32 "
        "FILE_ID_INFO query; see the note in stdio_guard._same_destination."
    ),
)
def test_stdout_is_diverted_when_stderr_is_the_same_pipe():
    """`2>&1` must not turn the guard into a silent no-op.

    Merging stderr into stdout is an ordinary way to launch a process. If the
    guard diverts fd 1 onto stderr in that case, it diverts the protocol channel
    onto itself: every diagnostic still lands on JSON-RPC while the guard looks
    installed.
    """
    script = textwrap.dedent(r"""
        import os, sys
        from arxiv_mcp_server.stdio_guard import protected_stdout
        with protected_stdout() as protocol:
            print("STRAY_WITH_MERGED_STDERR")
            os.write(1, b"RAW_WITH_MERGED_STDERR\n")
            protocol.write('{"jsonrpc":"2.0","id":7}\n')
            protocol.flush()
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # the aliasing case
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert result.stdout == '{"jsonrpc":"2.0","id":7}\n'


def test_a_vacated_fd_one_is_not_handed_to_the_protocol():
    """If the host moved stdout and closed fd 1, the protocol must not land there.

    `os.dup` returns the lowest free descriptor, so a vacant fd 1 would become
    the private protocol channel — exactly where raw writers and C extensions
    aim, reintroducing the corruption the guard exists to prevent.
    """
    result = _run_child(r"""
        moved = os.dup(1)                       # keep the real stdout alive
        os.close(1)                             # ...and vacate fd 1
        sys.stdout = os.fdopen(moved, "w")
        with protected_stdout() as protocol:
            os.write(1, b"RAW_TO_CONVENTIONAL_FD\n")
            protocol.write('{"jsonrpc":"2.0","id":8}\n')
            protocol.flush()
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":8}\n'


def test_host_output_between_sessions_reaches_real_stdout():
    """Buffered host output must not be swept into the diagnostic sink.

    Bytes written to stdout outside the guard are the host's own output. A
    later session must not flush that buffer after diverting fd 1, or writes
    made between two sessions silently vanish into stderr.
    """
    result = _run_child(r"""
        with protected_stdout() as protocol:
            protocol.write("FIRST\n"); protocol.flush()
        print("BETWEEN_SESSIONS")               # buffered, not yet flushed
        with protected_stdout() as protocol:
            protocol.write("SECOND\n"); protocol.flush()
        """)

    assert result.returncode == 0, result.stderr
    assert "BETWEEN_SESSIONS" in result.stdout
    assert "BETWEEN_SESSIONS" not in result.stderr


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Aliasing detection is POSIX-only. Windows anonymous pipes carry no "
        "filesystem identity — two independent pipes both report zero "
        "st_dev/st_ino — so _same_destination deliberately treats that as "
        "inconclusive rather than lose every diagnostic to the null device on "
        "the normal Windows arrangement. Closing this needs a Win32 "
        "FILE_ID_INFO query; see the note in stdio_guard._same_destination."
    ),
)
def test_an_occupied_fd_one_that_aliases_the_protocol_is_also_diverted():
    """A host may move sys.stdout to a dup and leave fd 1 open on the same pipe.

    Reserving fd 1 only when it is *free* leaves that case uncovered: raw
    `os.write(1, ...)`, C-extension output, and anything reaching
    `sys.__stdout__` still enter the JSON-RPC channel. Reproduced before the
    fix — the stray bytes landed on protocol stdout.
    """
    result = _run_child(r"""
        moved = os.dup(1)                       # fd 1 stays OPEN, same pipe
        sys.stdout = os.fdopen(moved, "w")
        with protected_stdout() as protocol:
            os.write(1, b"STRAY_VIA_OCCUPIED_FD1\n")
            protocol.write('{"jsonrpc":"2.0","id":10}\n')
            protocol.flush()
        os.write(1, b"HOST_FD1_RESTORED\n")
        """)

    assert result.returncode == 0, result.stderr
    assert "STRAY_VIA_OCCUPIED_FD1" not in result.stdout
    assert result.stdout.startswith('{"jsonrpc":"2.0","id":10}\n')
    assert "HOST_FD1_RESTORED" in result.stdout


def test_inconclusive_descriptor_identity_is_not_read_as_aliasing():
    """Windows anonymous pipes report a zero identity; two distinct pipes tie.

    Treating that tie as aliasing would send every diagnostic to the null device
    on the normal Windows arrangement, losing the MuPDF messages this guard
    exists to keep visible.
    """
    from arxiv_mcp_server.stdio_guard import _same_destination

    class _Zero:
        st_dev = 0
        st_ino = 0

    import os as _os

    real_fstat = _os.fstat
    try:
        _os.fstat = lambda fd: _Zero()
        assert _same_destination(1, 2) is False
    finally:
        _os.fstat = real_fstat


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="ctypes.CDLL(None) is POSIX-only; Windows has no single process-wide CRT",
)
def test_c_runtime_buffered_output_never_reaches_the_channel():
    """MuPDF prints through libc, and libc's buffer is invisible to Python.

    ``TextIOWrapper.flush`` drains Python's buffer alone. Bytes a C extension
    left in ``FILE *stdout`` survive the restore and land on JSON-RPC at the
    next flush — which is the failure this whole module exists to prevent, one
    layer lower than the Python-level cases above.
    """
    result = _run_child(r"""
        import ctypes
        libc = ctypes.CDLL(None)
        with protected_stdout() as protocol:
            libc.printf(b"C_BUFFERED_DIAGNOSTIC\n")   # deliberately not flushed
            protocol.write('{"jsonrpc":"2.0","id":11}\n')
            protocol.flush()
        libc.fflush(None)                             # the host's own later flush
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":11}\n'
    assert "C_BUFFERED_DIAGNOSTIC" in result.stderr


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "aliasing detection is POSIX-only: Windows anonymous pipes report a zero "
        "fstat identity, so _same_destination cannot see that stderr IS the "
        "protocol channel and the suppression this asserts never engages"
    ),
)
def test_the_guard_does_not_log_onto_the_channel_it_is_protecting():
    """With `2>&1` and debug logging on, the guard's own records are the leak.

    Every handler the guard could reach writes to fd 2, which in this
    arrangement *is* the JSON-RPC pipe — and stays so for the whole session, so
    deferring the record would not help either. Silence is the only safe
    output.

    Scoped to the guard's own records: import-time logging from ``mcp`` lands on
    the same pipe before the guard runs at all, and is not this module's to fix.
    """
    script = textwrap.dedent(r"""
        import logging, os, sys
        logging.basicConfig(level=logging.DEBUG)
        from arxiv_mcp_server.stdio_guard import protected_stdout
        with protected_stdout() as protocol:
            protocol.write('{"jsonrpc":"2.0","id":12}\n')
            protocol.flush()
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # the aliasing case
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "arxiv_mcp_server.stdio_guard" not in result.stdout
    assert result.stdout.endswith('{"jsonrpc":"2.0","id":12}\n')


def test_descriptor_inheritability_survives_the_round_trip():
    """`os.dup2` defaults to inheritable=True, so restoring can widen a host fd.

    An embedding host that handed us a private stdout would find it inherited
    by every later subprocess — the protocol channel leaking into unrelated
    children, silently and permanently.
    """
    result = _run_child(r"""
        moved = os.dup(1)                       # os.dup yields a private fd
        sys.stdout = open(moved, "w", closefd=False)
        before_moved = os.get_inheritable(moved)
        before_one = os.get_inheritable(1)
        with protected_stdout() as protocol:
            protocol.write('{"jsonrpc":"2.0","id":13}\n')
            protocol.flush()
        print(f"moved {before_moved}->{os.get_inheritable(moved)}", file=sys.stderr)
        print(f"one {before_one}->{os.get_inheritable(1)}", file=sys.stderr)
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":13}\n'
    assert "moved False->False" in result.stderr
    assert "one True->True" in result.stderr


@pytest.mark.skipif(
    sys.platform != "win32", reason="STD_OUTPUT_HANDLE exists only on Windows"
)
def test_the_win32_standard_output_handle_is_put_back():
    """Cleanup must restore the handle the host had, not one derived from fd 1.

    The Win32 slot is not a function of the descriptor table: a host that moved
    stdout can legitimately have STD_OUTPUT_HANDLE pointing elsewhere again.
    Repointing it at the restored descriptor redirects native code and every
    subprocess for the rest of the process's life.
    """
    result = _run_child(r"""
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetStdHandle.restype = ctypes.c_void_p
        k32.GetStdHandle.argtypes = (ctypes.c_int,)
        before = k32.GetStdHandle(-11)
        with protected_stdout() as protocol:
            protocol.write('{"jsonrpc":"2.0","id":14}\n')
            protocol.flush()
        print(f"handle_restored={before == k32.GetStdHandle(-11)}", file=sys.stderr)
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"jsonrpc":"2.0","id":14}\n'
    assert "handle_restored=True" in result.stderr
