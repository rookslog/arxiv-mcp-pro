"""Configuration settings for the arXiv MCP server."""

import sys
from importlib.metadata import version, PackageNotFoundError
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import logging

try:
    _PACKAGE_VERSION = version("arxiv-mcp-pro")
except PackageNotFoundError:
    _PACKAGE_VERSION = "0.0.0"

logger = logging.getLogger(__name__)

# Lazy shared arxiv client — created on first use, not at import time
_arxiv_client = None


def get_arxiv_client(page_size: int = 100):
    """Return a shared arxiv.Client instance, creating it on first call.

    The arxiv Python client fetches pages using its own page_size setting. If
    left at the library default of 100, even a small max_results request causes
    an upstream API URL with max_results=100. Keep the client page size aligned
    with the requested result count so small searches make small API requests.
    """
    global _arxiv_client
    if _arxiv_client is None or getattr(_arxiv_client, "page_size", None) != page_size:
        import arxiv

        _arxiv_client = arxiv.Client(page_size=page_size)
    return _arxiv_client


class Settings(BaseSettings):
    """Server configuration settings."""

    APP_NAME: str = "arxiv-mcp-pro"
    APP_VERSION: str = _PACKAGE_VERSION
    MAX_RESULTS: int = 50
    BATCH_SIZE: int = 20
    REQUEST_TIMEOUT: int = 60
    # Default cap on paper characters returned by read_paper / download_paper
    # when the caller omits `max_chars`. Uncapped whole-paper returns (~137k
    # chars observed in the field) overflow MCP clients' per-tool-output limits
    # and block the read path entirely; responses carry `is_truncated` /
    # `next_start`, so capped reads are discoverable and pageable. An explicit
    # `max_chars` always wins. `0` disables the default cap (legacy behavior:
    # omitting max_chars returns full content).
    CONTENT_DEFAULT_MAX_CHARS: int = 60000
    TRANSPORT: str = "stdio"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ALLOWED_HOSTS: str = ""
    ALLOWED_ORIGINS: str = ""
    CITATION_MAX_EDGES: int | None = None
    SEMANTIC_SCHOLAR_API_KEY: str | None = None
    # Minimum seconds between Semantic Scholar requests (0 = no pacing, the
    # default, preserving exact prior behavior). An authenticated S2 key grants
    # ~1 request/second across all endpoints; set this to ~1.1 to pace requests
    # proactively instead of bursting and relying on 429 retry/backoff.
    SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL: float = 0.0
    # Minimum seconds between arXiv API requests (arXiv asks for >= 3s per IP,
    # globally — not per process). The pacer coordinates sibling sessions on one
    # machine through a lock file in STORAGE_PATH, so parallel/multi-agent use on
    # a shared storage dir stays under the limit. `0` disables all arXiv pacing,
    # including the cross-process lock file. Multiple machines behind one IP
    # remain uncoordinated (see the README "Parallel / multi-agent use" note).
    ARXIV_MIN_REQUEST_INTERVAL: float = 3.0
    # Optional override for the `semantic_search` embedding model. Unset (the
    # default) means each backend uses its built-in default model: the
    # lightweight model2vec backend (`[pro]`) uses `minishlab/potion-retrieval-32M`;
    # the sentence-transformers backend (`[pro-st]`) uses
    # `sentence-transformers/all-MiniLM-L6-v2`. When set, this replaces the
    # default for whichever backend is active — it MUST be a model2vec /
    # static-model repo id for the model2vec backend, or an ST-compatible id for
    # the sentence-transformers backend. Changing the model changes the embedding
    # vectors (and often their dimension), which invalidates the local semantic
    # index, so run the `reindex` tool after changing it.
    EMBEDDING_MODEL: str | None = None
    model_config = SettingsConfigDict(extra="allow")

    @property
    def STORAGE_PATH(self) -> Path:
        """Get the resolved storage path and ensure it exists.

        Returns:
            Path: The absolute storage path.
        """
        path = (
            self._get_storage_path_from_args()
            or Path.home() / ".arxiv-mcp-server" / "papers"
        )
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_storage_path_from_args(self) -> Path | None:
        """Extract storage path from command line arguments.

        Returns:
            Path | None: The storage path if specified in arguments, None otherwise.
        """
        args = sys.argv[1:]

        # If not enough arguments
        if len(args) < 2:
            return None

        # Look for the --storage-path option
        try:
            storage_path_index = args.index("--storage-path")
        except ValueError:
            return None

        # Early return if --storage-path is the last argument
        if storage_path_index + 1 >= len(args):
            return None

        # Try to resolve the path
        try:
            path = Path(args[storage_path_index + 1])
            return path.resolve()
        except (TypeError, ValueError) as e:
            # TypeError: If the path argument is not string-like
            # ValueError: If the path string is malformed
            logger.warning(f"Invalid storage path format: {e}")
        except OSError as e:
            # OSError: If the path contains invalid characters or is too long
            logger.warning(f"Invalid storage path: {e}")

        return None
