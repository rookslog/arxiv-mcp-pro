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
2. point fd 1 itself at stderr.

Step 2 is what makes this robust. It catches writes from C extensions
(MuPDF is a C library) and from any reference to the original stdout object
captured before this ran, neither of which a Python-level ``sys.stdout``
reassignment would intercept. Diagnostics are not discarded — they land on
stderr, where an MCP client and systemd both already collect them.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import TextIO

logger = logging.getLogger(__name__)

__all__ = ["protect_stdout"]


def _fileno_or_none(stream: object) -> int | None:
    """Return a stream's file descriptor, or None if it does not have a real one."""
    try:
        fileno = stream.fileno()  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation, ValueError, OSError):
        return None
    return fileno if isinstance(fileno, int) and fileno >= 0 else None


def protect_stdout() -> TextIO:
    """Move the JSON-RPC channel off fd 1 and redirect fd 1 to stderr.

    Returns the stream the MCP transport must write to. After this call,
    ``print()``, ``sys.stdout.write()``, and C-level writes to fd 1 all go to
    stderr and can no longer corrupt the protocol.

    Degrades safely: when stdout or stderr has no real file descriptor — under
    pytest capture, or when the server is embedded in a host process — the
    descriptor swap is skipped and only the Python-level redirect is applied.
    The returned stream is always the correct one to write protocol bytes to.
    """
    original_stdout: TextIO = sys.stdout

    stdout_fd = _fileno_or_none(original_stdout)
    stderr_fd = _fileno_or_none(sys.stderr)

    if stdout_fd is None or stderr_fd is None:
        # No real descriptors to juggle. Do what we can at the Python level so
        # that print() still cannot reach the protocol stream.
        sys.stdout = sys.stderr
        logger.debug(
            "stdio guard: no usable file descriptors; applied Python-level redirect only"
        )
        return original_stdout

    try:
        original_stdout.flush()
    except (ValueError, OSError):  # pragma: no cover - already-closed stream
        pass

    # 1. Private copy of the protocol channel, before anything else can touch it.
    protected_fd = os.dup(stdout_fd)

    # 2. Anything that writes to fd 1 from here on lands on stderr instead.
    os.dup2(stderr_fd, stdout_fd)

    protected: TextIO = io.TextIOWrapper(
        io.FileIO(protected_fd, "wb", closefd=True),
        encoding="utf-8",
        newline="",
        write_through=True,
    )

    # 3. Python-level references follow suit, so `print()` is routed rather than
    #    merely redirected — this keeps stdout and stderr a single ordered stream.
    sys.stdout = sys.stderr

    logger.debug(
        "stdio guard: protocol channel moved to private fd %d; fd %d now points at stderr",
        protected_fd,
        stdout_fd,
    )
    return protected
