"""Protect the stdio JSON-RPC channel from stray writes to stdout.

Under the stdio transport, file descriptor 1 *is* the MCP protocol channel:
every byte on it must be JSON-RPC. Any library that writes to stdout —
directly, or through a reference to ``sys.stdout`` captured at import time —
injects non-JSON bytes into the middle of that stream. The client's decoder
then fails on the first stray character, the session desynchronises, and the
server is killed by the resulting broken pipe.

This is not hypothetical. PyMuPDF (pulled in by the ``pdf`` extra, via
``pymupdf4llm``) captures ``sys.stdout`` at import time::

    # pymupdf/__init__.py
    _g_out_message = sys.stdout
    def message(text):
        print(text, file=_g_out_message, flush=1)

Any MuPDF diagnostic — a malformed xref, a broken font, a recoverable parse
error — is printed straight onto the protocol channel.

Rather than chase each library, this module moves the protocol channel out of
harm's way *at the file-descriptor level*:

1. duplicate the original fd 1 to a private descriptor and hand that to the
   MCP transport, then
2. point fd 1 itself at stderr (or, when stderr is unusable, at the null
   device).

Step 2 is what makes this robust. It catches writes from C extensions
(MuPDF is a C library) and from any reference to the original stdout object
captured before this ran, neither of which a Python-level ``sys.stdout``
reassignment would intercept. Where stderr is available, diagnostics are not
discarded: they land there, and an MCP client and systemd both already collect
it.

The guard is a context manager because the redirect is process-global. A
server embedded in a host process must get its stdout back when the session
ends — otherwise the host's own output stays diverted, and a second session
would duplicate the already-redirected descriptor and answer onto stderr.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from typing import Iterator, TextIO

logger = logging.getLogger(__name__)

__all__ = ["protected_stdout"]

_STD_OUTPUT_HANDLE = -11
_STDOUT_FD = 1


def _fileno_or_none(stream: object) -> int | None:
    """Return a stream's *live* file descriptor, or None.

    The liveness check matters. A stream object keeps reporting the descriptor
    number it was built with even after that descriptor is closed, and the
    number is then free to be handed to the next ``os.dup``. Trusting a stale
    number is how a guard ends up copying the protocol channel onto fd 1, so
    confirm the descriptor is actually open before believing it.
    """
    try:
        fileno = stream.fileno()  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation, ValueError, OSError):
        return None
    if not isinstance(fileno, int) or fileno < 0:
        return None
    try:
        os.fstat(fileno)
    except OSError:
        return None
    return fileno


def _is_free(fd: int) -> bool:
    """True when *fd* is not currently open."""
    try:
        os.fstat(fd)
    except OSError:
        return True
    return False


def _same_destination(a: int, b: int) -> bool:
    """True when two descriptors refer to the same open file.

    `2>&1` and `stderr=subprocess.STDOUT` are ordinary ways to launch a
    process, and both leave stderr pointing at the very pipe carrying JSON-RPC.
    Diverting stdout onto stderr would then be a no-op — the guard would look
    installed and protect nothing.
    """
    try:
        sa, sb = os.fstat(a), os.fstat(b)
    except OSError:  # pragma: no cover - descriptor closed underneath us
        return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def _sync_win32_std_handle(stdout_fd: int) -> None:
    """Repoint the Win32 STD_OUTPUT_HANDLE slot at whatever fd 1 now refers to.

    ``os.dup2`` rewrites the C runtime descriptor table but leaves the Win32
    standard-handle slot alone. Native code and subprocesses that reach stdout
    through ``GetStdHandle`` would otherwise keep writing to the original
    JSON-RPC pipe. Best effort: a failure here costs the Win32-level half of
    the guard, not the CRT-level half, so it is logged rather than raised.
    """
    if sys.platform != "win32":  # pragma: no cover - platform-specific
        return
    try:  # pragma: no cover - exercised only on Windows
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(stdout_fd)
        if not ctypes.windll.kernel32.SetStdHandle(_STD_OUTPUT_HANDLE, handle):
            raise OSError(ctypes.get_last_error())
    except Exception as exc:  # pragma: no cover - never fatal
        logger.debug("stdio guard: could not repoint STD_OUTPUT_HANDLE: %r", exc)


@contextlib.contextmanager
def protected_stdout() -> Iterator[TextIO]:
    """Move the JSON-RPC channel off fd 1 for the duration of the block.

    Yields the stream the MCP transport must write to. Inside the block,
    ``print()``, ``sys.stdout.write()``, and C-level writes to fd 1 all go to
    stderr — or to the null device when stderr has no descriptor of its own —
    and can no longer corrupt the protocol. On exit fd 1 and ``sys.stdout`` are
    both restored, so an embedding host gets its stdout back.

    Degrades safely when stdout has no real file descriptor (pytest capture, or
    an embedding host that replaced the object): the descriptor swap is skipped
    and only the Python-level redirect is applied. The yielded stream is always
    the correct one to write protocol bytes to.
    """
    original_stdout: TextIO = sys.stdout
    stdout_fd = _fileno_or_none(original_stdout)

    if stdout_fd is None:
        # Nothing to juggle at the descriptor level. Do what we can so that
        # print() still cannot reach the protocol stream.
        sys.stdout = sys.stderr
        logger.debug("stdio guard: no stdout descriptor; Python-level redirect only")
        try:
            yield original_stdout
        finally:
            sys.stdout = original_stdout
        return

    # 1. Settle where diagnostics will go BEFORE duplicating anything. Order is
    #    load-bearing: os.dup hands out the lowest free descriptor, so a closed
    #    stderr would have its number reclaimed by the protocol channel below,
    #    and a later check against the stale number would divert fd 1 onto the
    #    protocol itself. Opening the fallback sink first also means that when
    #    fd 2 *is* free, the sink claims it rather than the protocol.
    #
    #    Note what this deliberately does NOT do: `sys.stderr` having no
    #    usable fileno does not prove descriptor 2 is dead. A host that swaps
    #    in an io.StringIO leaves fd 2 perfectly alive, and clobbering it —
    #    then closing it on the way out — would destroy a descriptor this
    #    guard does not own. The sink is only ever used as a dup2 *source*.
    # Flush what is already buffered while fd 1 still points at the host's real
    # stdout. These bytes were written before the guard existed, so they are the
    # host's own output and belong there — deferring the flush past the
    # diversion would silently redirect legitimate output into the sink, and a
    # write between two sessions would vanish.
    with contextlib.suppress(ValueError, OSError):
        original_stdout.flush()

    stderr_fd = _fileno_or_none(sys.stderr)
    if stderr_fd is not None and _same_destination(stderr_fd, stdout_fd):
        # stderr IS the protocol channel; it cannot also be the sink.
        logger.debug("stdio guard: stderr aliases stdout; falling back to os.devnull")
        stderr_fd = None
    sink_fd: int | None = None
    if stderr_fd is None:
        sink_fd = os.open(os.devnull, os.O_WRONLY)

    # Reserve the conventional stdout descriptor if the host has vacated it.
    # Otherwise `os.dup` below, which hands out the lowest free number, would
    # put the private protocol channel on fd 1 — precisely where raw writers
    # and C extensions aim.
    reserved_stdout_fd = False
    if stdout_fd != _STDOUT_FD and _is_free(_STDOUT_FD):
        placeholder = os.open(os.devnull, os.O_WRONLY)
        if placeholder != _STDOUT_FD:
            os.dup2(placeholder, _STDOUT_FD)
            os.close(placeholder)
        reserved_stdout_fd = True
        logger.debug("stdio guard: reserved vacant fd %d", _STDOUT_FD)

    # 2. Private copy of the protocol channel, before anything else can touch it.
    protected_fd = os.dup(stdout_fd)
    protected: TextIO = io.TextIOWrapper(
        io.FileIO(protected_fd, "wb", closefd=True),
        encoding="utf-8",
        newline="",
        write_through=True,
    )

    # 3. fd 1 now refers to the diagnostic sink — stderr where it is usable,
    #    the null device otherwise.
    os.dup2(stderr_fd if stderr_fd is not None else sink_fd, stdout_fd)
    _sync_win32_std_handle(stdout_fd)

    # 4. Python-level references follow suit, so print() is routed rather than
    #    merely redirected — this keeps stdout and stderr a single ordered
    #    stream. With no usable stderr there is nowhere better to send it than
    #    the original object, which now writes to the diverted fd 1.
    if stderr_fd is not None:
        sys.stdout = sys.stderr

    logger.debug(
        "stdio guard: protocol channel moved to private fd %d; fd %d diverted",
        protected_fd,
        stdout_fd,
    )

    try:
        yield protected
    finally:
        # Order matters here too. A library holding a pre-guard reference to
        # stdout may have written without flushing — normal when stdout is a
        # pipe — leaving diagnostics sitting in that wrapper's buffer. Flush it
        # while fd 1 still points at the sink; flushing after the restore would
        # empty those bytes straight onto the JSON-RPC channel.
        with contextlib.suppress(ValueError, OSError):
            original_stdout.flush()
        with contextlib.suppress(ValueError, OSError):
            protected.flush()
        try:
            os.dup2(protected_fd, stdout_fd)
            _sync_win32_std_handle(stdout_fd)
        except OSError as exc:  # pragma: no cover - descriptor already reclaimed
            logger.debug("stdio guard: could not restore fd %d: %r", stdout_fd, exc)
        finally:
            sys.stdout = original_stdout
            if sink_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(sink_fd)
            if reserved_stdout_fd:
                with contextlib.suppress(OSError):
                    os.close(_STDOUT_FD)
            with contextlib.suppress(ValueError, OSError):
                protected.close()
