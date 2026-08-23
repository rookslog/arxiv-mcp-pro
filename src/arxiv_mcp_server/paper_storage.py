"""Helpers for mapping arXiv IDs to flat local cache paths."""

from pathlib import Path
from urllib.parse import quote, unquote


def paper_id_to_stem(paper_id: str) -> str:
    """Encode an arXiv ID as a filesystem-safe, reversible filename stem."""
    return quote(paper_id, safe="")


def paper_id_from_stem(stem: str) -> str:
    """Decode a filename stem produced by :func:`paper_id_to_stem`.

    A stem that this module would not itself have written is returned
    unchanged, so it stays whatever it looked like on disk and is judged as
    such. Decoding it unconditionally invents papers: `2401%2E12345.md` is
    not a file `paper_id_to_stem` can produce, but `unquote` turns it into
    the perfectly valid-looking `2401.12345`, which `list_papers` then
    advertises and `read_paper` cannot open — it re-encodes the ID to
    `2401.12345.md`, a different file that does not exist. One directory
    entry becomes a paper the server offers and then denies.
    """
    decoded = unquote(stem)
    return decoded if paper_id_to_stem(decoded) == stem else stem


def paper_path(storage_path: str | Path, paper_id: str, suffix: str = ".md") -> Path:
    """Return the flat cache path for an arXiv paper ID."""
    return Path(storage_path) / f"{paper_id_to_stem(paper_id)}{suffix}"
