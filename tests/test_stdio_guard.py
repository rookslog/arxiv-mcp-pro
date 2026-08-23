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
from arxiv_mcp_server.stdio_guard import protect_stdout


def _run_child(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a child process with real pipes for stdout and stderr."""
    script = textwrap.dedent("""
        import os, sys
        from arxiv_mcp_server.stdio_guard import protect_stdout
        """) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_stray_python_writes_do_not_reach_the_protocol_channel():
    """print() after the guard goes to stderr; only protocol bytes reach stdout."""
    result = _run_child("""
        protocol = protect_stdout()
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
        protocol = protect_stdout()
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
        protocol = protect_stdout()
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
        protocol = protect_stdout()
        print("keep me visible")
        protocol.write("{}\\n")
        protocol.flush()
        """)

    assert result.returncode == 0, result.stderr
    assert "keep me visible" in result.stderr
    assert "keep me visible" not in result.stdout


def test_degrades_safely_without_real_descriptors(monkeypatch):
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

    returned = protect_stdout()

    assert returned is sentinel_stdout
    assert sys.stdout is sentinel_stderr


@pytest.mark.asyncio
async def test_run_stdio_hands_the_protected_stream_to_the_transport():
    """_run_stdio must give the transport the protected stream, not sys.stdout."""
    protected = MagicMock(name="protected_stdout")
    wrapped = MagicMock(name="wrapped")
    transport = MagicMock()
    transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    transport.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(server_module, "protect_stdout", return_value=protected) as guard,
        patch.object(server_module.anyio, "wrap_file", return_value=wrapped) as wrap,
        patch.object(
            server_module, "stdio_server", return_value=transport
        ) as stdio_server,
        patch.object(server_module.server, "run", new_callable=AsyncMock) as run,
    ):
        await server_module._run_stdio()

    guard.assert_called_once_with()
    wrap.assert_called_once_with(protected)
    stdio_server.assert_called_once_with(stdout=wrapped)
    run.assert_awaited_once()
