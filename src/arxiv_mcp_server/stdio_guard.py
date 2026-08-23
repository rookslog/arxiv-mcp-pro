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
from typing import Any, Iterator, TextIO

logger = logging.getLogger(__name__)

__all__ = ["protected_stdout"]

_STD_OUTPUT_HANDLE = -11
_STDOUT_FD = 1

_libc: Any = None
_libc_resolved = False

# Distinct from None, which is a *legitimate* STD_OUTPUT_HANDLE value meaning
# the host has no standard-output handle at all. Conflating "could not read it"
# with "read it, and it was NULL" would hand a host a handle on the way out
# that it never had.
_HANDLE_UNAVAILABLE = object()


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


def _inheritable(fd: int) -> bool:
    """Whether *fd* is inherited by child processes; False if unknowable."""
    try:
        return os.get_inheritable(fd)
    except OSError:  # pragma: no cover - descriptor closed underneath us
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

    # Windows gives anonymous pipes no filesystem identity — two independent
    # pipes both report zero. Reading that as "aliased" would send every
    # diagnostic to the null device on the normal Windows arrangement, losing
    # exactly the MuPDF messages this guard exists to keep visible. Treat a
    # zero identity as inconclusive and assume the descriptors are distinct.
    #
    # The consequence is honest and bounded: aliasing detection is verified on
    # POSIX only. Under `2>&1` on Windows the guard degrades to what it was
    # before this check existed — diagnostics go to stderr, which in that
    # arrangement is the protocol channel, and the guard's own debug records go
    # with them if the host has DEBUG logging on. Closing that needs a Win32
    # handle-identity call (FILE_ID_INFO on a pipe handle), which is untestable
    # from here; three tests carry a matching skipif for the same reason.
    if (sa.st_dev, sa.st_ino) == (0, 0) or (sb.st_dev, sb.st_ino) == (0, 0):
        return False

    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def _flush_c_stdio(*, quiet: bool = False) -> None:
    """Drain the C runtime's own stdout buffer.

    ``TextIOWrapper.flush`` empties Python's buffer and nothing else. A C
    extension that printed with ``printf``/``fputs`` leaves its bytes in
    libc's ``FILE *stdout`` buffer, which Python cannot see and which stays
    unflushed while stdout is a pipe. MuPDF is a C library, so this is exactly
    the output the guard exists to catch: left in that buffer while the sink is
    installed, the bytes surface on the JSON-RPC channel the moment the
    descriptor is restored, or at interpreter shutdown.

    ``fflush(NULL)`` flushes every open output stream, which is what we want —
    the diagnostic must land wherever fd 1 currently points, not later.

    Best effort. Windows has no single process-wide C runtime — an extension
    may link its own copy — so a flush there covers the runtimes we can name,
    not necessarily the one that did the printing.
    """
    global _libc, _libc_resolved

    if not _libc_resolved:
        _libc_resolved = True
        try:
            import ctypes

            if sys.platform == "win32":  # pragma: no cover - platform-specific
                for name in ("api-ms-win-crt-stdio-l1-1-0", "ucrtbase", "msvcrt"):
                    try:
                        _libc = ctypes.CDLL(name)
                        break
                    except OSError:
                        continue
            else:
                _libc = ctypes.CDLL(None)
        except Exception as exc:  # pragma: no cover - never fatal
            _libc = None
            if not quiet:
                logger.debug("stdio guard: no C runtime to flush: %r", exc)

    if _libc is None:
        return
    try:
        _libc.fflush(None)
    except Exception as exc:  # pragma: no cover - never fatal
        if not quiet:
            logger.debug("stdio guard: could not flush C stdio: %r", exc)


def _win32_kernel32() -> Any:  # pragma: no cover - platform-specific
    """kernel32 with pointer-width types on the standard-handle calls.

    Left to its defaults ctypes treats a HANDLE as a C ``int``, which truncates
    on 64-bit Windows: the value round-trips wrongly, or raises outright once a
    handle exceeds 2**31.
    """
    import ctypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetStdHandle.restype = ctypes.c_void_p
    k32.GetStdHandle.argtypes = (ctypes.c_int,)
    k32.SetStdHandle.restype = ctypes.c_int
    k32.SetStdHandle.argtypes = (ctypes.c_int, ctypes.c_void_p)
    return k32


def _win32_stdout_handle(*, quiet: bool = False) -> Any:
    """The process's current STD_OUTPUT_HANDLE.

    Returns `_HANDLE_UNAVAILABLE` when the value could not be read, which is
    not the same as reading it and finding NULL — a host may legitimately have
    no standard-output handle, and `ctypes` reports that as `None`.
    """
    if sys.platform != "win32":
        return _HANDLE_UNAVAILABLE
    try:  # pragma: no cover - exercised only on Windows
        return _win32_kernel32().GetStdHandle(_STD_OUTPUT_HANDLE)
    except Exception as exc:  # pragma: no cover - never fatal
        if not quiet:
            logger.debug("stdio guard: could not read STD_OUTPUT_HANDLE: %r", exc)
        return _HANDLE_UNAVAILABLE


def _win32_handle_for_fd(fd: int, *, quiet: bool = False) -> Any:
    """The Win32 handle backing *fd*, or `_HANDLE_UNAVAILABLE`."""
    if sys.platform != "win32":
        return _HANDLE_UNAVAILABLE
    try:  # pragma: no cover - exercised only on Windows
        import msvcrt

        return msvcrt.get_osfhandle(fd)
    except Exception as exc:  # pragma: no cover - never fatal
        if not quiet:
            logger.debug("stdio guard: no handle for fd %d: %r", fd, exc)
        return _HANDLE_UNAVAILABLE


def _win32_set_stdout_handle(handle: Any, *, quiet: bool = False) -> None:
    """Point the Win32 STD_OUTPUT_HANDLE slot at *handle*.

    `None` is a value, not an absence: it sets the slot to NULL, which is what
    restoring a host that had no standard-output handle requires.
    `_HANDLE_UNAVAILABLE` is the absence, and does nothing.

    ``os.dup2`` rewrites the C runtime descriptor table but leaves this slot
    alone. Native code and subprocesses that reach stdout through
    ``GetStdHandle`` would otherwise keep writing to the original JSON-RPC
    pipe. Best effort: a failure here costs the Win32-level half of the guard,
    not the CRT-level half, so it is logged rather than raised.
    """
    if sys.platform != "win32" or handle is _HANDLE_UNAVAILABLE:
        return
    try:  # pragma: no cover - exercised only on Windows
        import ctypes

        if not _win32_kernel32().SetStdHandle(_STD_OUTPUT_HANDLE, handle):
            raise OSError(ctypes.get_last_error())
    except Exception as exc:  # pragma: no cover - never fatal
        if not quiet:
            logger.debug("stdio guard: could not set STD_OUTPUT_HANDLE: %r", exc)


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
    stderr_fd = _fileno_or_none(sys.stderr)
    stderr_is_protocol = stderr_fd is not None and _same_destination(
        stderr_fd, stdout_fd
    )

    def _diag(message: str, *args: object) -> None:
        """Log a guard diagnostic, unless doing so would corrupt the channel.

        When stderr aliases stdout there is no destination the guard can log
        to without risking the protocol stream: the usual handler writes to
        fd 2, which *is* the JSON-RPC pipe, and stays so for the whole session
        — deferring the record past the descriptor swap would not help. A
        debug line in a rare launch arrangement is worth less than the
        session, so the guard goes quiet instead.
        """
        if stderr_is_protocol:
            return
        logger.debug(message, *args)

    # Flush Python's buffer while fd 1 still points at the host's real stdout.
    # These bytes were written before the guard existed, so they are the host's
    # own output and belong there — deferring this past the diversion would
    # silently redirect legitimate output into the sink, and a write between two
    # sessions would vanish.
    #
    # The C runtime's buffer is deliberately NOT drained here, and the asymmetry
    # is the whole point. Under the stdio transport fd 1 is the protocol channel
    # from process start, and a native extension imported before this ran — the
    # server imports its tools, and the pdf tool pulls in PyMuPDF — may already
    # have a diagnostic sitting in libc's buffer. Flushing it now would push it
    # straight onto JSON-RPC, the guard causing the exact corruption it exists
    # to prevent. It is drained below instead, once the sink is in place.
    # Python's layer needs no such care: pymupdf prints with `flush=1`, so
    # nothing accumulates there to begin with.
    with contextlib.suppress(ValueError, OSError):
        original_stdout.flush()

    if stderr_is_protocol:
        # stderr IS the protocol channel; it cannot also be the sink.
        stderr_fd = None

    sink_fd: int | None = None
    reserved_stdout_fd = False
    alias_saved_fd: int | None = None
    alias_inheritable = False
    protected_fd: int | None = None
    protected: TextIO | None = None
    wrapper_owns_fd = False

    # An embedding host may have handed us a non-inheritable stdout — that is
    # what os.dup() returns — and os.dup2() defaults to inheritable=True. Left
    # alone, the guard would silently widen the descriptor on the way out and
    # every later subprocess would inherit the protocol channel. Carry the
    # host's flag through every swap instead.
    stdout_inheritable = _inheritable(stdout_fd)

    # The Win32 standard-handle slot is not derivable from fd 1: a host that
    # moved stdout elsewhere can legitimately have STD_OUTPUT_HANDLE pointing
    # somewhere else again. Save the exact handle now so cleanup can put that
    # one back rather than inventing a new one from the restored descriptor.
    original_win32_stdout = _win32_stdout_handle(quiet=stderr_is_protocol)

    # Descriptor acquisition is all-or-nothing. Without this, a failure partway
    # through — EMFILE on the dup, say — would leave the host's fd 1 claimed by
    # the null device and the sink leaked, with no `finally` yet in scope to
    # release them, because the context never entered.
    try:
        if stderr_fd is None:
            sink_fd = os.open(os.devnull, os.O_WRONLY)
        sink = stderr_fd if stderr_fd is not None else sink_fd

        if stdout_fd != _STDOUT_FD:
            if _is_free(_STDOUT_FD):
                # Reserve the conventional descriptor the host has vacated,
                # so the `os.dup` below — which returns the lowest free number
                # — cannot put the private protocol channel on fd 1, precisely
                # where raw writers and C extensions aim.
                placeholder = os.open(os.devnull, os.O_WRONLY)
                if placeholder != _STDOUT_FD:
                    os.dup2(placeholder, _STDOUT_FD, inheritable=False)
                    os.close(placeholder)
                reserved_stdout_fd = True
                _diag("stdio guard: reserved vacant fd %d", _STDOUT_FD)
            elif _same_destination(_STDOUT_FD, stdout_fd):
                # fd 1 is still open and still points at the protocol pipe, so
                # diverting only the moved descriptor would leave raw writes to
                # fd 1 — and anything reaching sys.__stdout__ — on the channel.
                alias_inheritable = _inheritable(_STDOUT_FD)
                alias_saved_fd = os.dup(_STDOUT_FD)
                os.dup2(sink, _STDOUT_FD, inheritable=alias_inheritable)
                _diag("stdio guard: diverted fd %d aliasing the protocol", _STDOUT_FD)

        protected_fd = os.dup(stdout_fd)
        protected = io.TextIOWrapper(
            io.FileIO(protected_fd, "wb", closefd=True),
            encoding="utf-8",
            newline="",
            write_through=True,
        )
        wrapper_owns_fd = True  # closing the wrapper now closes the descriptor

        os.dup2(sink, stdout_fd, inheritable=stdout_inheritable)
        _win32_set_stdout_handle(
            _win32_handle_for_fd(stdout_fd, quiet=stderr_is_protocol),
            quiet=stderr_is_protocol,
        )
        # Now that fd 1 lands in the sink, it is safe to drain whatever a native
        # extension buffered before the guard existed. Doing it here rather than
        # at teardown also means an operator watching stderr sees the diagnostic
        # near when it happened, not at session end.
        _flush_c_stdio(quiet=stderr_is_protocol)
    except BaseException:
        if protected is not None:
            with contextlib.suppress(ValueError, OSError):
                protected.close()
        elif protected_fd is not None and not wrapper_owns_fd:
            with contextlib.suppress(OSError):
                os.close(protected_fd)
        if alias_saved_fd is not None:
            with contextlib.suppress(OSError):
                os.dup2(alias_saved_fd, _STDOUT_FD, inheritable=alias_inheritable)
                os.close(alias_saved_fd)
        if reserved_stdout_fd:
            with contextlib.suppress(OSError):
                os.close(_STDOUT_FD)
        if sink_fd is not None:
            with contextlib.suppress(OSError):
                os.close(sink_fd)
        raise

    # Python-level references follow suit, so print() is routed rather than
    # merely redirected — this keeps stdout and stderr a single ordered stream.
    # With no usable stderr there is nowhere better than the original object,
    # which now writes to the diverted fd.
    if stderr_fd is not None:
        sys.stdout = sys.stderr

    _diag("stdio guard: protocol channel moved off fd %d", stdout_fd)

    try:
        yield protected
    finally:
        # Order matters here too. A library holding a pre-guard reference to
        # stdout may have written without flushing — normal when stdout is a
        # pipe — leaving diagnostics sitting in that wrapper's buffer. Flush it
        # while fd 1 still points at the sink; flushing after the restore would
        # empty those bytes straight onto the JSON-RPC channel. The same is
        # true one layer down, in the C runtime's buffer, where a native
        # extension's printf output waits — and that layer is the one MuPDF
        # writes through.
        with contextlib.suppress(ValueError, OSError):
            original_stdout.flush()
        with contextlib.suppress(ValueError, OSError):
            protected.flush()
        _flush_c_stdio(quiet=stderr_is_protocol)
        try:
            os.dup2(protected_fd, stdout_fd, inheritable=stdout_inheritable)
            # Put back the handle the host actually had. Falling back to the
            # restored descriptor's handle is only for the case where the
            # original could not be read at all.
            _win32_set_stdout_handle(
                (
                    original_win32_stdout
                    if original_win32_stdout is not _HANDLE_UNAVAILABLE
                    else _win32_handle_for_fd(stdout_fd, quiet=stderr_is_protocol)
                ),
                quiet=stderr_is_protocol,
            )
        except OSError as exc:  # pragma: no cover - descriptor already reclaimed
            _diag("stdio guard: could not restore fd %d: %r", stdout_fd, exc)
        finally:
            sys.stdout = original_stdout
            if sink_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(sink_fd)
            if alias_saved_fd is not None:
                with contextlib.suppress(OSError):
                    os.dup2(alias_saved_fd, _STDOUT_FD, inheritable=alias_inheritable)
                    os.close(alias_saved_fd)
            if reserved_stdout_fd:
                with contextlib.suppress(OSError):
                    os.close(_STDOUT_FD)
            with contextlib.suppress(ValueError, OSError):
                protected.close()
