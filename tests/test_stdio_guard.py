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
