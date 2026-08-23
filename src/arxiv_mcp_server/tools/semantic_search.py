"""Semantic search and indexing tools for the arXiv MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import arxiv
from pydantic import Field
import mcp.types as types
from mcp.types import ToolAnnotations

from ..schemas import ToolInput, schema_from_model
from ..config import Settings
from .arxiv_pacing import pace_arxiv_request_sync, record_arxiv_request
from .list_papers import is_valid_arxiv_id
from ..paper_storage import paper_id_from_stem

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled gracefully in runtime checks
    np = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled gracefully in runtime checks
    SentenceTransformer = None  # type: ignore[assignment]

logger = logging.getLogger("arxiv-mcp-pro")
settings = Settings()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DB_NAME = "semantic_index.db"

_model: Optional[Any] = None

# Guards a running `reindex` against concurrent `semantic_search` reads. With
# `clear_existing=True`, `rebuild_index` commits `DELETE FROM semantic_index`
# then slowly re-indexes off the event loop (B20); without this lock a search
# landing mid-rebuild would read a just-cleared / partial corpus. Lazy + module
# level so it can rebind across event loops in tests, mirroring
# download.py's `_get_index_semaphore` (asyncio primitives only bind to a loop
# on their contended slow path, so uncontended callers are loop-agnostic).
_reindex_lock: Optional[asyncio.Lock] = None


def _get_reindex_lock() -> asyncio.Lock:
    """Return the module-level reindex lock, creating it lazily."""
    global _reindex_lock
    if _reindex_lock is None:
        _reindex_lock = asyncio.Lock()
    return _reindex_lock


@dataclass
class IndexedPaper:
    """Stored paper payload used for similarity ranking."""

    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published: str
    score: float


class ReindexInput(ToolInput):
    """Arguments for the `reindex` tool."""

    clear_existing: bool = Field(
        default=True,
        description=("If true, clear the existing index before rebuilding."),
    )


class SemanticSearchInput(ToolInput):
    """Arguments for the `semantic_search` tool."""

    compact: Optional[bool] = Field(
        default=None,
        description=(
            "Drop the full `abstract` from each result to cut token cost; all other fields (id, title, authors, categories, published, score, resource_uri) are kept. Omit for full output."
        ),
    )
    max_results: int = Field(
        default=10,
        ge=0,
        description=("Maximum number of results to return (default: 10)."),
    )
    offset: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Pagination offset into the ranked results (page size = max_results). Send `offset: 0` for page one WITH a cursor (`total_available`/`next_offset`), then follow `next_offset` for later pages. Omit `offset` entirely (with `compact` also unset) for legacy unpaged output and no cursor."
        ),
    )
    paper_id: Optional[str] = Field(
        default=None,
        description=("Find papers semantically similar to this arXiv paper ID."),
    )
    query: Optional[str] = Field(
        default=None, description=("Free-text semantic query.")
    )


semantic_search_tool = types.Tool(
    name="semantic_search",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    description=(
        "Semantic similarity search over papers you have already downloaded locally via download_paper. "
        "Supports free-text queries (e.g. 'attention mechanisms for long sequences') or finding papers "
        "similar to a given paper_id. "
        "IMPORTANT: only searches your local downloaded collection — will return empty results if no papers "
        "have been downloaded yet. Use search_papers to find papers on arXiv, then download_paper to add "
        "them to the local index before using this tool. "
        "Opt-in pagination: set `offset` to page through ranked results (page size = max_results); "
        "set `compact` to drop the full abstract from each result and cut token cost. When either is set, "
        "the response adds `offset`/`total_available`/`next_offset`. Omit both for full, unpaged output. "
        'Requires the [pro] extra: pip install "arxiv-mcp-pro[pro]" '
        '(from a source checkout: uv pip install -e ".[pro]").'
    ),
    inputSchema=schema_from_model(SemanticSearchInput),
)


reindex_tool = types.Tool(
    name="reindex",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, openWorldHint=False
    ),
    description="Rebuild the local semantic index for downloaded papers.",
    inputSchema=schema_from_model(ReindexInput),
)


def _dependency_error() -> Optional[str]:
    """Return a friendly dependency error if pro packages are missing."""
    if np is None or SentenceTransformer is None:
        return (
            "Pro feature dependency missing. Install with: "
            '`pip install "arxiv-mcp-pro[pro]"` '
            '(from a source checkout: `uv pip install -e ".[pro]"`)'
        )
    return None


def _db_path() -> Path:
    """Return the semantic index SQLite path."""
    return Path(settings.STORAGE_PATH) / INDEX_DB_NAME


def _connect() -> sqlite3.Connection:
    """Open SQLite connection and ensure schema exists."""
    conn = sqlite3.connect(_db_path())
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_index (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                categories_json TEXT NOT NULL,
                published TEXT,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
        conn.commit()
    except Exception:
        # Schema init failed after the connection opened — close it rather than
        # leak it. Callers use `with closing(_connect())`, which never receives
        # (and so never closes) the connection if _connect raises mid-setup.
        conn.close()
        raise
    return conn


def _get_model() -> Any:
    """Load the sentence-transformers model lazily."""
    global _model
    if _model is None:
        logger.info("Loading semantic embedding model %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, silent=True)
    return _model


def _embed_text(text: str) -> Any:
    """Create embedding vector for a text payload."""
    model = _get_model()
    return model.encode(text or "", convert_to_numpy=True, normalize_embeddings=True)


def _upsert_index_record(
    paper_id: str,
    title: str,
    abstract: str,
    authors: List[str],
    categories: List[str],
    published: str = "",
) -> bool:
    """Insert or update an index record for a paper."""
    dependency_error = _dependency_error()
    if dependency_error:
        logger.warning(dependency_error)
        return False

    embedding = _embed_text(abstract)
    embedding_array = np.asarray(embedding, dtype=np.float32)

    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO semantic_index (
                paper_id, title, abstract, authors_json, categories_json,
                published, embedding, embedding_dim, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title=excluded.title,
                abstract=excluded.abstract,
                authors_json=excluded.authors_json,
                categories_json=excluded.categories_json,
                published=excluded.published,
                embedding=excluded.embedding,
                embedding_dim=excluded.embedding_dim,
                updated_at=excluded.updated_at
            """,
            (
                paper_id,
                title,
                abstract,
                json.dumps(authors),
                json.dumps(categories),
                published,
                embedding_array.tobytes(),
                int(embedding_array.shape[0]),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    return True


def index_paper_by_id(paper_id: str) -> bool:
    """Fetch arXiv metadata by ID and add/update it in the semantic index.

    Blocking (network + embedding) — run it off the event loop
    (``asyncio.to_thread``); the async handlers in this module and download.py
    do. Pacing: the metadata fetch goes through the sync cross-process pacer
    (B20) — pacing is the pacer's job, not any client-internal delay. The
    ``arxiv.Client()`` is created per call because this function runs in worker
    threads, so no ``requests.Session`` is shared across threads; the shared
    ``get_arxiv_client()`` instance is used by the foreground paths.
    """
    try:
        client = arxiv.Client()
        pace_arxiv_request_sync()
        try:
            paper = next(client.results(arxiv.Search(id_list=[paper_id])))
        finally:
            # Even a failed attempt hit the network; record it so sibling
            # lanes pace off the same clock.
            record_arxiv_request()
    except StopIteration:
        logger.warning("Could not index paper %s: not found on arXiv", paper_id)
        return False
    except Exception as exc:
        logger.error("Could not fetch metadata for %s: %s", paper_id, exc)
        return False

    return index_paper_from_result(paper)


def index_paper_from_result(paper: Any) -> bool:
    """Index a paper from an arxiv.Result-like object."""
    try:
        paper_id = paper.get_short_id()
        title = paper.title or ""
        abstract = paper.summary or ""
        authors = [author.name for author in getattr(paper, "authors", [])]
        categories = list(getattr(paper, "categories", []) or [])
        published = ""
        if getattr(paper, "published", None) is not None:
            published = paper.published.isoformat()

        if not abstract.strip():
            logger.warning(
                "Skipping semantic indexing for %s: empty abstract", paper_id
            )
            return False

        return _upsert_index_record(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=categories,
            published=published,
        )
    except Exception as exc:
        logger.error("Failed indexing paper from result: %s", exc)
        return False


def _load_vectors(exclude_paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all vectors (optionally excluding one paper)."""
    with closing(_connect()) as conn:
        if exclude_paper_id:
            rows = conn.execute(
                "SELECT * FROM semantic_index WHERE paper_id != ?", (exclude_paper_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM semantic_index").fetchall()

    vectors: List[Dict[str, Any]] = []
    for row in rows:
        vector = np.frombuffer(
            row["embedding"], dtype=np.float32, count=row["embedding_dim"]
        )
        vectors.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "authors": json.loads(row["authors_json"]),
                "categories": json.loads(row["categories_json"]),
                "published": row["published"] or "",
                "vector": vector,
            }
        )
    return vectors


def _rank_by_similarity(
    query_vector: Any,
    candidates: List[Dict[str, Any]],
    max_results: int,
    offset: int = 0,
) -> List[IndexedPaper]:
    """Compute cosine similarity (normalized vectors) and rank results."""
    if not candidates:
        return []

    matrix = np.vstack([candidate["vector"] for candidate in candidates])
    similarities = matrix @ np.asarray(query_vector, dtype=np.float32)

    ranked_indices = np.argsort(similarities)[::-1][offset : offset + max_results]
    ranked_results: List[IndexedPaper] = []

    for idx in ranked_indices:
        candidate = candidates[int(idx)]
        ranked_results.append(
            IndexedPaper(
                paper_id=candidate["paper_id"],
                title=candidate["title"],
                abstract=candidate["abstract"],
                authors=candidate["authors"],
                categories=candidate["categories"],
                published=candidate["published"],
                score=float(similarities[int(idx)]),
            )
        )

    return ranked_results


def _get_indexed_paper_vector(paper_id: str) -> Optional[Any]:
    """Fetch an indexed vector for a specific paper."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT embedding, embedding_dim FROM semantic_index WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()

    if row is None:
        return None

    return np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_dim"])


def rebuild_index(clear_existing: bool = True) -> Dict[str, Any]:
    """Rebuild semantic index from downloaded markdown papers."""
    dependency_error = _dependency_error()
    if dependency_error:
        return {"status": "error", "message": dependency_error}

    paper_ids = sorted(
        paper_id_from_stem(p.stem)
        for p in Path(settings.STORAGE_PATH).glob("*.md")
        if is_valid_arxiv_id(paper_id_from_stem(p.stem))
    )

    if clear_existing:
        with closing(_connect()) as conn:
            conn.execute("DELETE FROM semantic_index")
            conn.commit()

    indexed = 0
    failed: List[str] = []

    for paper_id in paper_ids:
        success = index_paper_by_id(paper_id)
        if success:
            indexed += 1
        else:
            failed.append(paper_id)

    return {
        "status": "success",
        "indexed": indexed,
        "failed": failed,
        "total_local_papers": len(paper_ids),
    }


async def handle_reindex(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle reindex tool calls."""
    try:
        clear_existing = bool(arguments.get("clear_existing", True))
        # Off the event loop: rebuild_index makes one paced arXiv call per
        # local paper (N × the pacing interval under B20) plus embedding work.
        # Run inline, that would freeze every other tool for minutes.
        # Hold _reindex_lock so a concurrent semantic_search waits for the
        # rebuild rather than reading a just-cleared / partially-rebuilt index.
        #
        # Tie the lock's release to the WORKER's completion, not this
        # coroutine's. Cancelling `await asyncio.to_thread(...)` (client
        # disconnect / request timeout) does NOT stop the worker thread — a
        # plain `async with` would exit and release the lock while the thread
        # is still mid-`DELETE FROM semantic_index` + repopulate, letting a
        # semantic_search acquire the lock and read the cleared/partial corpus
        # (codex P2). Acquire manually; release only from the worker's
        # done-callback (runs on the loop, fires exactly once). asyncio.shield
        # keeps the worker uncancelled; on cancellation CancelledError
        # propagates out of the handler (BaseException on 3.11 — NOT caught by
        # `except Exception` below), while the thread runs on holding the lock.
        lock = _get_reindex_lock()
        await lock.acquire()
        try:
            worker = asyncio.ensure_future(
                asyncio.to_thread(rebuild_index, clear_existing)
            )
        except BaseException:
            lock.release()
            raise
        worker.add_done_callback(lambda _t: lock.release())
        result = await asyncio.shield(worker)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as exc:
        logger.error("Reindex failed: %s", exc)
        return [types.TextContent(type="text", text=f"Error: {str(exc)}")]


async def handle_semantic_search(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle semantic search queries and similar-paper lookups."""
    try:
        dependency_error = _dependency_error()
        if dependency_error:
            return [types.TextContent(type="text", text=f"Error: {dependency_error}")]

        query = (arguments.get("query") or "").strip()
        paper_id = (arguments.get("paper_id") or "").strip()
        # Clamp to [0, MAX_RESULTS]: a negative max_results would otherwise feed a
        # negative slice bound into _rank_by_similarity (offset:offset+max_results),
        # producing surprising pages/cursors. 0 stays valid (empty page).
        max_results = min(
            max(0, int(arguments.get("max_results", 10))), settings.MAX_RESULTS
        )
        # Capture whether `offset` was explicitly provided (even as 0): an
        # explicit offset opts into paginated mode so the client gets cursor
        # metadata to page forward. `offset` omitted (or null) stays legacy.
        offset_arg = arguments.get("offset")
        offset = max(0, int(offset_arg or 0))
        # Strict boolean (like citation_graph's `compact`/`counts_only`): a string
        # such as "false" is truthy under bool(), which would silently drop every
        # abstract and switch into paginated mode.
        compact = arguments.get("compact") is True

        if not query and not paper_id:
            return [
                types.TextContent(
                    type="text",
                    text="Error: Provide either `query` or `paper_id` for semantic_search.",
                )
            ]

        # Serialize the read/rank against a running reindex: searches wait for a
        # running rebuild rather than reading a just-cleared index (pre-B20, the
        # inline rebuild blocked the loop and serialized these de facto; the lock
        # restores those observable semantics while keeping OTHER tools
        # responsive). Holding it across the query-mode _embed_text too is
        # harmless and keeps the critical section a single block. Unlike
        # reindex, a plain `async with` is safe here: a cancelled search
        # releases the lock with no orphan worker mutating shared state — the
        # only write on this path is a single index_paper_by_id upsert, never a
        # clear+rebuild, so an early release cannot expose a cleared index.
        async with _get_reindex_lock():
            if paper_id:
                query_vector = _get_indexed_paper_vector(paper_id)
                if query_vector is None:
                    logger.info(
                        "Paper %s not indexed yet, attempting to fetch and index",
                        paper_id,
                    )
                    # Off the event loop: blocking network fetch, paced (B20).
                    if not await asyncio.to_thread(index_paper_by_id, paper_id):
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Error: Could not index source paper {paper_id}.",
                            )
                        ]
                    query_vector = _get_indexed_paper_vector(paper_id)

                candidates = _load_vectors(exclude_paper_id=paper_id)
                mode = "similar_to_paper"
                query_payload = paper_id
            else:
                query_vector = _embed_text(query)
                candidates = _load_vectors()
                mode = "semantic_query"
                query_payload = query

            ranked = _rank_by_similarity(
                query_vector, candidates, max_results=max_results, offset=offset
            )
            total_available = len(candidates)

        papers = []
        for paper in ranked:
            paper_dict = {
                "id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "categories": paper.categories,
                "published": paper.published,
                "score": round(paper.score, 6),
                "resource_uri": f"arxiv://{paper.paper_id}",
            }
            if compact:
                # pop (not del) so a future change making `abstract` conditional
                # can't raise a KeyError that the broad except below would swallow.
                paper_dict.pop("abstract", None)
            papers.append(paper_dict)

        # PAGINATED MODE = `offset` explicitly provided (even 0) or `compact`.
        # Only the default — `offset` omitted AND not compact — leaves the
        # response byte-for-byte identical to the legacy shape. An explicit
        # `offset: 0` is a deliberate opt-in: unlike citation_graph there is no
        # separate `limit` param to trigger pagination while starting at 0.
        paginated = offset_arg is not None or compact
        response = {
            "mode": mode,
            "query": query_payload,
            "total_results": len(ranked),
        }
        if paginated:
            next_offset = offset + len(ranked)
            response["offset"] = offset
            response["total_available"] = total_available
            # Only emit a cursor when this page actually returned results. An empty
            # page (e.g. max_results=0, or offset past the end) must not point a
            # client back at the same offset, or it loops forever.
            response["next_offset"] = (
                next_offset if ranked and next_offset < total_available else None
            )
        response["papers"] = papers

        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
    except Exception as exc:
        logger.error("Semantic search failed: %s", exc)
        return [types.TextContent(type="text", text=f"Error: {str(exc)}")]
