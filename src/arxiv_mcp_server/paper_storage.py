"""Helpers for mapping arXiv IDs to flat local cache paths."""

from pathlib import Path
from urllib.parse import quote, unquote


def paper_id_to_stem(paper_id: str) -> str:
    """Encode an arXiv ID as a filesystem-safe, reversible filename stem."""
    return quote(paper_id, safe="")


def paper_id_from_stem(stem: str) -> str:
    """Decode a filename stem produced by :func:`paper_id_to_stem`."""
    return unquote(stem)


def paper_path(storage_path: str | Path, paper_id: str, suffix: str = ".md") -> Path:
    """Return the flat cache path for an arXiv paper ID."""
    return Path(storage_path) / f"{paper_id_to_stem(paper_id)}{suffix}"
