"""Tests for semantic search and reindex tools."""

import asyncio
import json
import threading
from pathlib import Path

import pytest

from arxiv_mcp_server.tools import semantic_search as semantic_module

np = pytest.importorskip("numpy")


class DummyModel:
    """Deterministic embedding model for tests."""

    def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
        vector = np.array(
            [
                float("transformer" in text.lower()),
                float("vision" in text.lower()),
                float("graph" in text.lower()),
            ],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


@pytest.fixture
def semantic_test_env(monkeypatch, temp_storage_path):
    """Configure semantic search module to use a temporary index and dummy model."""
    monkeypatch.setattr(
        semantic_module.settings,
        "_get_storage_path_from_args",
        lambda: Path(temp_storage_path),
    )
    monkeypatch.setattr(semantic_module, "SentenceTransformer", object)
    monkeypatch.setattr(semantic_module, "_get_model", lambda: DummyModel())
    # R5: hermetic against a real EMBEDDING_MODEL in the environment — pin the
    # override to None so tests see the default per-backend model name unless a
    # test sets it explicitly. Settings() reads env vars, so without this a
    # developer's exported EMBEDDING_MODEL would change identities and break the
    # B23 compat-guard assertions.
    monkeypatch.setattr(semantic_module.settings, "EMBEDDING_MODEL", None)
    semantic_module._model = None


@pytest.mark.asyncio
async def test_semantic_search_free_text(semantic_test_env):
    """Semantic text query should rank closest abstract first."""
    semantic_module._upsert_index_record(
        paper_id="2401.00001",
        title="Vision Transformers",
        abstract="transformer model for vision",
        authors=["Author 1"],
        categories=["cs.CV"],
    )
    semantic_module._upsert_index_record(
        paper_id="2401.00002",
        title="Graph Methods",
        abstract="graph neural network approach",
        authors=["Author 2"],
        categories=["cs.LG"],
    )

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 2}
    )

    payload = json.loads(response[0].text)
    assert payload["total_results"] == 2
    assert payload["papers"][0]["id"] == "2401.00001"


@pytest.mark.asyncio
async def test_semantic_search_by_paper_id(semantic_test_env):
    """similar-to-paper mode excludes the source paper from results."""
    semantic_module._upsert_index_record(
        paper_id="2402.00001",
        title="Transformer Baselines",
        abstract="transformer pretraining method",
        authors=["Author 1"],
        categories=["cs.LG"],
    )
    semantic_module._upsert_index_record(
        paper_id="2402.00002",
        title="Vision Transformer Variant",
        abstract="vision transformer architecture",
        authors=["Author 2"],
        categories=["cs.CV"],
    )

    response = await semantic_module.handle_semantic_search(
        {"paper_id": "2402.00001", "max_results": 3}
    )

    payload = json.loads(response[0].text)
    assert payload["mode"] == "similar_to_paper"
    assert all(p["id"] != "2402.00001" for p in payload["papers"])


def _index_three_papers():
    """Index three papers with deterministic, distinct DummyModel rankings.

    Against the query "vision transformer" (DummyModel vector ~ [1, 1, 0]):
      - 2403.00001 "transformer model for vision" -> [1, 1, 0] -> rank 1
      - 2403.00002 "vision graph methods"         -> [0, 1, 1] -> rank 2
      - 2403.00003 "graph neural network"         -> [0, 0, 1] -> rank 3
    """
    semantic_module._upsert_index_record(
        paper_id="2403.00001",
        title="Vision Transformers",
        abstract="transformer model for vision",
        authors=["Author 1"],
        categories=["cs.CV"],
    )
    semantic_module._upsert_index_record(
        paper_id="2403.00002",
        title="Vision Graphs",
        abstract="vision graph methods",
        authors=["Author 2"],
        categories=["cs.LG"],
    )
    semantic_module._upsert_index_record(
        paper_id="2403.00003",
        title="Graph Networks",
        abstract="graph neural network",
        authors=["Author 3"],
        categories=["cs.LG"],
    )


@pytest.mark.asyncio
async def test_semantic_search_compact_drops_abstract(semantic_test_env):
    """compact=True omits the abstract key and adds the pagination metadata."""
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 3, "compact": True}
    )

    payload = json.loads(response[0].text)
    assert payload["total_results"] == 3
    for paper in payload["papers"]:
        assert "abstract" not in paper
        assert "id" in paper
        assert "title" in paper
        assert "score" in paper
    # Paginated mode metadata is present.
    assert payload["offset"] == 0
    assert payload["total_available"] == 3
    # Last page (3 of 3 returned) -> no further page.
    assert payload["next_offset"] is None


@pytest.mark.asyncio
async def test_semantic_search_offset_pages(semantic_test_env):
    """offset=1 returns the 2nd-ranked paper; next_offset advances correctly."""
    _index_three_papers()

    full = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 3}
    )
    full_payload = json.loads(full[0].text)
    second_ranked_id = full_payload["papers"][1]["id"]

    page = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 1, "offset": 1}
    )
    page_payload = json.loads(page[0].text)

    assert page_payload["total_results"] == 1
    assert page_payload["papers"][0]["id"] == second_ranked_id
    assert page_payload["offset"] == 1
    assert page_payload["total_available"] == 3
    # offset(1) + returned(1) = 2 < total_available(3) -> next page at 2.
    assert page_payload["next_offset"] == 2


@pytest.mark.asyncio
async def test_semantic_search_default_output_unchanged(semantic_test_env):
    """No offset/compact: abstracts present, no pagination metadata (legacy)."""
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 3}
    )

    payload = json.loads(response[0].text)
    assert set(payload.keys()) == {"mode", "query", "total_results", "papers"}
    for paper in payload["papers"]:
        assert "abstract" in paper
    assert "offset" not in payload
    assert "total_available" not in payload
    assert "next_offset" not in payload


@pytest.mark.asyncio
async def test_semantic_search_explicit_offset_zero_is_paginated(semantic_test_env):
    """Explicit offset=0 (no compact) opts into pagination: cursor metadata is
    present so a client can discover/follow page 2, while abstracts are still
    included (only `compact` drops them). Distinguishes an explicit offset:0 from
    an omitted offset (codex P2 — honor explicit offset=0 pagination requests)."""
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 1, "offset": 0}
    )

    payload = json.loads(response[0].text)
    # Paginated shape: cursor metadata present even though offset is 0...
    assert payload["offset"] == 0
    assert payload["total_available"] == 3
    # offset(0) + returned(1) = 1 < total_available(3) -> next page at 1.
    assert payload["next_offset"] == 1
    # ...but not compact, so abstracts remain.
    for paper in payload["papers"]:
        assert "abstract" in paper


@pytest.mark.asyncio
async def test_semantic_search_negative_max_results_clamped(semantic_test_env):
    """A negative max_results is clamped to 0 (empty page), not passed as a
    negative slice bound — no crash, no nonsensical cursor (codex cross-vendor
    finding: max_results was only upper-clamped)."""
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": -5, "offset": 0}
    )

    payload = json.loads(response[0].text)
    assert payload["total_results"] == 0
    assert payload["papers"] == []
    # offset:0 is explicit -> paginated, but an empty page emits no cursor.
    assert payload["offset"] == 0
    assert payload["next_offset"] is None


def test_connect_closes_connection_on_schema_init_failure(
    monkeypatch, semantic_test_env
):
    """If schema setup raises, _connect closes the connection it opened instead
    of leaking it (codex cross-vendor finding: closing() at the call sites does
    not cover a failure inside _connect itself)."""
    import sqlite3

    closed = {"value": False}

    class _FakeConn:
        row_factory = None

        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("schema init boom")

        def commit(self):  # pragma: no cover - not reached
            pass

        def close(self):
            closed["value"] = True

    monkeypatch.setattr(semantic_module.sqlite3, "connect", lambda *a, **k: _FakeConn())

    with pytest.raises(sqlite3.OperationalError):
        semantic_module._connect()
    assert closed["value"] is True


@pytest.mark.asyncio
async def test_semantic_search_next_offset_null_at_end(semantic_test_env):
    """An offset that reaches the last page yields next_offset is None."""
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 1, "offset": 2}
    )

    payload = json.loads(response[0].text)
    assert payload["total_results"] == 1
    assert payload["offset"] == 2
    assert payload["total_available"] == 3
    # offset(2) + returned(1) = 3, not < total_available(3) -> end of results.
    assert payload["next_offset"] is None


@pytest.mark.asyncio
async def test_semantic_search_offset_past_end(semantic_test_env):
    """An offset beyond total_available returns an empty page, not an error or loop."""
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 3, "offset": 99}
    )

    payload = json.loads(response[0].text)
    assert payload["total_results"] == 0
    assert payload["papers"] == []
    assert payload["offset"] == 99
    assert payload["total_available"] == 3
    # offset(99) is not < total_available(3) -> no next page (no infinite paging).
    assert payload["next_offset"] is None


@pytest.mark.asyncio
async def test_semantic_search_compact_with_offset(semantic_test_env):
    """compact and offset combine: dropped abstract AND correct pagination cursor."""
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 1, "offset": 1, "compact": True}
    )

    payload = json.loads(response[0].text)
    assert payload["total_results"] == 1
    assert "abstract" not in payload["papers"][0]
    assert payload["papers"][0]["id"]  # still carries identity fields
    assert payload["offset"] == 1
    assert payload["total_available"] == 3
    # offset(1) + returned(1) = 2 < total_available(3) -> next page at 2.
    assert payload["next_offset"] == 2


@pytest.mark.asyncio
async def test_semantic_search_compact_strict_boolean(semantic_test_env):
    """A truthy non-bool like the string 'false' must NOT enable compact (codex P2).

    bool('false') is True; mirroring citation_graph's `is True` guard keeps a lax
    client from silently dropping abstracts and flipping into paginated mode.
    """
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 3, "compact": "false"}
    )

    payload = json.loads(response[0].text)
    # Not compact, not paginated -> the legacy shape with abstracts present.
    assert set(payload.keys()) == {"mode", "query", "total_results", "papers"}
    assert all("abstract" in p for p in payload["papers"])


@pytest.mark.asyncio
async def test_semantic_search_zero_page_size_no_cursor_loop(semantic_test_env):
    """max_results=0 in paginated mode yields an empty page with next_offset None (codex P2).

    Otherwise next_offset == offset and a client following the cursor loops forever
    on the same empty page.
    """
    _index_three_papers()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 0, "offset": 0, "compact": True}
    )

    payload = json.loads(response[0].text)
    assert payload["total_results"] == 0
    assert payload["papers"] == []
    # The empty page must not advertise a self-referential cursor.
    assert payload["next_offset"] is None


@pytest.mark.asyncio
async def test_reindex_uses_local_markdown_ids(
    monkeypatch, semantic_test_env, temp_storage_path
):
    """Reindex should walk local markdown files and attempt indexing each ID."""
    Path(temp_storage_path, "2301.00001.md").write_text("paper", encoding="utf-8")
    Path(temp_storage_path, "2301.00002.md").write_text("paper", encoding="utf-8")

    indexed_ids = []

    def _mock_index(paper_id):
        indexed_ids.append(paper_id)
        return True

    monkeypatch.setattr(semantic_module, "index_paper_by_id", _mock_index)

    response = await semantic_module.handle_reindex({"clear_existing": True})

    payload = json.loads(response[0].text)
    assert payload["status"] == "success"
    assert set(indexed_ids) == {"2301.00001", "2301.00002"}


@pytest.mark.asyncio
async def test_semantic_search_waits_for_running_reindex(
    semantic_test_env, monkeypatch
):
    """A semantic_search issued while a reindex is running must block on the
    shared _reindex_lock until the rebuild finishes, rather than reading a
    just-cleared / partial index (codex P2 — reindex/read race).

    The fake rebuild runs in a worker thread (handle_reindex offloads it via
    asyncio.to_thread). It signals `started` (rebuild has entered, and
    handle_reindex is holding the lock), then blocks on `release` — a
    threading.Event because the block happens off the event loop.
    """
    started = threading.Event()
    release = threading.Event()

    def _fake_rebuild(clear_existing=True):
        started.set()
        # Hold the reindex lock (via handle_reindex) until the test releases us.
        release.wait(timeout=5)
        return {
            "status": "success",
            "indexed": 0,
            "failed": [],
            "total_local_papers": 0,
        }

    monkeypatch.setattr(semantic_module, "rebuild_index", _fake_rebuild)
    # Keep the search path off the DB/model: deterministic embed + empty corpus.
    monkeypatch.setattr(
        semantic_module, "_embed_text", lambda text: np.zeros(3, dtype=np.float32)
    )
    monkeypatch.setattr(semantic_module, "_load_vectors", lambda *a, **k: [])

    reindex_task = asyncio.create_task(semantic_module.handle_reindex({}))
    # Wait until the rebuild is actually running inside the lock.
    assert await asyncio.to_thread(started.wait, 5) is True

    search_task = asyncio.create_task(
        semantic_module.handle_semantic_search({"query": "x"})
    )
    # Give the search a chance to run; it must be parked on the reindex lock.
    await asyncio.sleep(0.05)
    assert search_task.done() is False

    # Let the rebuild finish; the search should then acquire the lock and return.
    release.set()
    await reindex_task
    response = await search_task

    payload = json.loads(response[0].text)
    assert payload["mode"] == "semantic_query"
    assert payload["total_results"] == 0
    assert payload["papers"] == []


@pytest.mark.asyncio
async def test_reindex_lock_held_until_worker_finishes_after_cancel(
    semantic_test_env, monkeypatch
):
    """If handle_reindex is CANCELLED (client disconnect / timeout) while the
    rebuild worker thread is still running, the lock must stay held until the
    WORKER finishes — not be released when the cancelled coroutine unwinds.
    Otherwise a semantic_search could acquire the lock and read a just-cleared /
    partial index mid-rebuild (codex P2).

    Cancelling `await asyncio.to_thread(rebuild_index, ...)` does not stop the
    thread; the fix ties the release to the worker's done-callback, so a search
    issued after the cancel must park on the lock until the worker returns.
    """
    # The module-global _reindex_lock binds to an event loop on its contended
    # slow path. Any earlier lock-contending test (e.g. the sibling
    # test_semantic_search_waits_for_running_reindex) leaves it bound to that
    # test's now-closed loop; contending it again here on pytest-asyncio's fresh
    # per-test loop would raise "bound to a different event loop". Reset it so
    # _get_reindex_lock() mints a fresh lock on THIS loop (mirrors the fixture's
    # `_model = None` reset). Purely a test-harness artifact — production runs a
    # single long-lived loop, so the lock binds once.
    monkeypatch.setattr(semantic_module, "_reindex_lock", None)

    started = threading.Event()
    release = threading.Event()

    def _fake_rebuild(clear_existing=True):
        started.set()
        # Keep the worker (and thus the lock) alive past the coroutine cancel.
        release.wait(timeout=5)
        return {
            "status": "success",
            "indexed": 0,
            "failed": [],
            "total_local_papers": 0,
        }

    monkeypatch.setattr(semantic_module, "rebuild_index", _fake_rebuild)
    # Keep the search path off the DB/model: deterministic embed + empty corpus.
    monkeypatch.setattr(
        semantic_module, "_embed_text", lambda text: np.zeros(3, dtype=np.float32)
    )
    monkeypatch.setattr(semantic_module, "_load_vectors", lambda *a, **k: [])

    reindex_task = asyncio.create_task(semantic_module.handle_reindex({}))
    # Wait until the rebuild worker is actually running (lock is held).
    assert await asyncio.to_thread(started.wait, 5) is True

    # Cancel the handler coroutine. The worker thread keeps running; the fix
    # must keep the lock held until that worker completes.
    reindex_task.cancel()
    await asyncio.sleep(0.05)
    assert reindex_task.cancelled() or reindex_task.done()

    # A search issued now must park on the still-held lock — the cancelled
    # handler must NOT have released it early. THIS is the assertion the fix
    # protects; the plain `async with` form fails right here.
    search_task = asyncio.create_task(
        semantic_module.handle_semantic_search({"query": "x"})
    )
    await asyncio.sleep(0.05)
    assert search_task.done() is False

    # Let the worker finish; its done-callback releases the lock and the search
    # then proceeds. Awaiting the search to completion also drains the worker
    # (the release happens on the worker's completion), so no task leaks.
    release.set()
    response = await asyncio.wait_for(search_task, 5)

    payload = json.loads(response[0].text)
    assert payload["mode"] == "semantic_query"
    assert payload["total_results"] == 0
    assert payload["papers"] == []


# ---------------------------------------------------------------------------
# index_paper_by_id arXiv pacing wiring (B20)
#
# The metadata fetch is paced through the SYNC cross-process pacer and recorded
# after (finally-path), using a fresh arxiv.Client() per call (thread-confined —
# NOT the shared get_arxiv_client(), whose requests.Session must not cross
# threads). pace_arxiv_request_sync / record_arxiv_request are patched on the
# module namespace — the import style is `from .arxiv_pacing import
# pace_arxiv_request_sync, record_arxiv_request`, so they live as
# `semantic_module.<name>`; the client is patched via `semantic_module.arxiv.Client`.
# ---------------------------------------------------------------------------


def test_index_paper_by_id_paces_before_fetch_records_after(monkeypatch):
    """B20: index_paper_by_id calls the sync pacer BEFORE the arXiv fetch and
    record_arxiv_request AFTER, via a fresh per-call arxiv.Client()."""
    events = []
    monkeypatch.setattr(
        semantic_module, "pace_arxiv_request_sync", lambda: events.append("pace")
    )
    monkeypatch.setattr(
        semantic_module, "record_arxiv_request", lambda: events.append("record")
    )
    # Isolate the pacing wiring from the embedding/DB path.
    monkeypatch.setattr(semantic_module, "index_paper_from_result", lambda paper: True)

    mock_paper = object()

    class _Client:
        def results(self, search):
            events.append("fetch")
            return iter([mock_paper])

    client = _Client()
    monkeypatch.setattr(semantic_module.arxiv, "Client", lambda *a, **k: client)

    assert semantic_module.index_paper_by_id("2401.00001") is True
    # Paced before the fetch, recorded after — exact order.
    assert events == ["pace", "fetch", "record"]


def test_index_paper_by_id_records_even_when_fetch_raises(monkeypatch):
    """B20 finally-path: a failed fetch still records so sibling lanes pace off
    the same clock; the handler swallows the error and returns False."""
    events = []
    monkeypatch.setattr(
        semantic_module, "pace_arxiv_request_sync", lambda: events.append("pace")
    )
    monkeypatch.setattr(
        semantic_module, "record_arxiv_request", lambda: events.append("record")
    )

    class _Client:
        def results(self, search):
            events.append("fetch")
            raise RuntimeError("arxiv unreachable")

    monkeypatch.setattr(semantic_module.arxiv, "Client", lambda *a, **k: _Client())

    assert semantic_module.index_paper_by_id("2401.00001") is False
    assert events == ["pace", "fetch", "record"]


# ---------------------------------------------------------------------------
# B23: model2vec / sentence-transformers backend abstraction + index-compat guard
# ---------------------------------------------------------------------------


class DummyStaticModel:
    """model2vec-style static model: `.encode(text)` returns a fixed vector.

    The vector is deliberately NON-unit so a test can prove `_embed_text` adds NO
    normalization layer of its own on the model2vec path (F5: normalization is
    enforced at LOAD time via `from_pretrained(..., normalize=True)`, not by
    `_embed_text`; if `_embed_text` re-normalized, these values would collapse to
    a unit vector and a zero vector would become NaN).
    """

    def __init__(self, vector):
        self._vector = vector

    def encode(self, text):
        return np.array(self._vector, dtype=np.float32)


def _reset_model_globals(monkeypatch):
    """Force _get_model to reload on the next call (mirrors the fixture reset)."""
    monkeypatch.setattr(semantic_module, "_model", None)
    monkeypatch.setattr(semantic_module, "_model_backend", None)
    monkeypatch.setattr(semantic_module, "_model_name", None)


def test_active_backend_prefers_sentence_transformers(monkeypatch):
    """When both backends import, sentence-transformers wins (upgraders unaffected)."""
    monkeypatch.setattr(semantic_module, "SentenceTransformer", object)
    monkeypatch.setattr(semantic_module, "StaticModel", object)
    assert semantic_module._active_backend() == "sentence-transformers"
    assert semantic_module._dependency_error() is None


def test_active_backend_model2vec_when_st_absent(monkeypatch):
    """ST missing but model2vec present -> model2vec backend, no dependency error."""
    monkeypatch.setattr(semantic_module, "SentenceTransformer", None)
    monkeypatch.setattr(semantic_module, "StaticModel", object)
    assert semantic_module._active_backend() == "model2vec"
    assert semantic_module._dependency_error() is None


def test_active_backend_none_gives_two_extra_dependency_error(monkeypatch):
    """No backend -> _active_backend None and the error names BOTH install extras."""
    monkeypatch.setattr(semantic_module, "SentenceTransformer", None)
    monkeypatch.setattr(semantic_module, "StaticModel", None)
    assert semantic_module._active_backend() is None
    err = semantic_module._dependency_error()
    assert err is not None
    assert "[pro]" in err
    assert "[pro-st]" in err


def test_model2vec_embed_path_does_not_renormalize(monkeypatch):
    """model2vec backend: _get_model loads via from_pretrained with normalize=True
    (F5); _embed_text returns the model's vector untouched (adds no normalization
    layer of its own)."""
    captured = {}

    class _StaticModelFactory:
        @staticmethod
        def from_pretrained(name, normalize=False, force_download=True):
            captured["name"] = name
            captured["normalize"] = normalize
            captured["force_download"] = force_download
            return DummyStaticModel([3.0, 4.0])  # norm 5 -> unit iff re-normalized

    monkeypatch.setattr(semantic_module, "SentenceTransformer", None)
    monkeypatch.setattr(semantic_module, "StaticModel", _StaticModelFactory)
    monkeypatch.setattr(semantic_module.settings, "EMBEDDING_MODEL", None)
    _reset_model_globals(monkeypatch)

    vector = semantic_module._embed_text("hello")

    # Loaded through the model2vec path with the default model name AND the
    # cosine-safety kwarg (F5) AND the cache-friendly force_download=False (R3).
    assert captured["name"] == "minishlab/potion-retrieval-32M"
    assert captured["normalize"] is True
    assert captured["force_download"] is False
    assert isinstance(semantic_module._get_model(), DummyStaticModel)
    # Returned untouched — NOT collapsed to a unit vector by _embed_text.
    assert np.allclose(vector, np.array([3.0, 4.0], dtype=np.float32))
    assert vector.dtype == np.float32


def test_embedding_model_override_model2vec(monkeypatch):
    """EMBEDDING_MODEL overrides the model2vec default passed to from_pretrained."""
    captured = {}

    class _StaticModelFactory:
        @staticmethod
        def from_pretrained(name, normalize=False, force_download=True):
            captured["name"] = name
            captured["normalize"] = normalize
            return DummyStaticModel([1.0, 0.0])

    monkeypatch.setattr(semantic_module, "SentenceTransformer", None)
    monkeypatch.setattr(semantic_module, "StaticModel", _StaticModelFactory)
    monkeypatch.setattr(
        semantic_module.settings, "EMBEDDING_MODEL", "myorg/custom-static"
    )
    _reset_model_globals(monkeypatch)

    semantic_module._get_model()
    assert captured["name"] == "myorg/custom-static"
    assert captured["normalize"] is True


def test_embedding_model_override_sentence_transformers(monkeypatch):
    """EMBEDDING_MODEL overrides the ST default; the ST path keeps silent=True."""
    captured = {}

    class _STFactory:
        def __init__(self, name, silent=False):
            captured["name"] = name
            captured["silent"] = silent

        def encode(self, *args, **kwargs):
            return np.zeros(3, dtype=np.float32)

    monkeypatch.setattr(semantic_module, "SentenceTransformer", _STFactory)
    monkeypatch.setattr(semantic_module, "StaticModel", None)
    monkeypatch.setattr(semantic_module.settings, "EMBEDDING_MODEL", "myorg/custom-st")
    _reset_model_globals(monkeypatch)

    semantic_module._get_model()
    assert captured["name"] == "myorg/custom-st"
    assert captured["silent"] is True


@pytest.mark.asyncio
async def test_dim_mismatch_returns_reindex_error(semantic_test_env, monkeypatch):
    """A query vector whose dimension differs from the stored vectors trips the
    friendly reindex guard BEFORE _rank_by_similarity's `matrix @ query_vector`
    can raise a bare numpy shape error (B23)."""
    _index_three_papers()  # dim-3 vectors, index_meta stamped (ST, dim 3)

    # Simulate the active model producing a different dimension than the index.
    monkeypatch.setattr(
        semantic_module, "_embed_text", lambda text: np.zeros(4, dtype=np.float32)
    )

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer"}
    )
    text = response[0].text
    assert "Error:" in text
    assert "reindex" in text.lower()
    # The guard fired, not a raw numpy shape/matmul error.
    assert "matmul" not in text.lower()
    assert "shapes" not in text.lower()


@pytest.mark.asyncio
async def test_model_mismatch_returns_reindex_error(semantic_test_env, monkeypatch):
    """A model switch with the SAME dimension is caught by the meta model-identity
    check (not the dimension check) — fires the reindex guard."""
    _index_three_papers()  # meta stamped with the default ST model

    # Same backend + dimension, different model name -> meta no longer matches.
    monkeypatch.setattr(semantic_module.settings, "EMBEDDING_MODEL", "different/model")

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer"}
    )
    text = response[0].text
    assert "Error:" in text
    assert "reindex" in text.lower()


@pytest.mark.asyncio
async def test_legacy_index_without_meta_searches_normally(semantic_test_env):
    """A legacy pre-B23 index (rows present, no index_meta) with a matching
    dimension must NOT trip the guard (no false positive)."""
    _index_three_papers()

    # Strip index_meta to mimic an index built before B23 added the table.
    conn = semantic_module._connect()
    try:
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 3}
    )
    payload = json.loads(response[0].text)
    assert payload["total_results"] == 3
    assert payload["papers"][0]["id"] == "2403.00001"


@pytest.mark.asyncio
async def test_reindex_refreshes_index_meta(
    semantic_test_env, monkeypatch, temp_storage_path
):
    """After a clear_existing rebuild, index_meta reflects the active backend/model."""
    Path(temp_storage_path, "2404.00001.md").write_text("paper", encoding="utf-8")

    def _mock_index(paper_id):
        return semantic_module._upsert_index_record(
            paper_id=paper_id,
            title="Title",
            abstract="transformer vision study",
            authors=[],
            categories=[],
        )

    monkeypatch.setattr(semantic_module, "index_paper_by_id", _mock_index)

    await semantic_module.handle_reindex({"clear_existing": True})

    meta = semantic_module._read_index_meta()
    assert (
        meta["embedding_model"]
        == "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2"
    )
    assert meta["embedding_dim"] == "3"


@pytest.mark.asyncio
async def test_reindex_no_clear_refuses_dim_mismatch(semantic_test_env, monkeypatch):
    """reindex(clear_existing=false) refuses to append vectors of a different
    dimension onto the existing index (mixing dims would corrupt/crash ranking)."""
    # Reset the module lock onto this test's loop (mirrors the cancel test): a
    # sibling lock-contending test may have bound it to a now-closed loop.
    monkeypatch.setattr(semantic_module, "_reindex_lock", None)
    _index_three_papers()  # dim-3 rows

    # Active model now produces dim-4 vectors -> appending would mix dimensions.
    monkeypatch.setattr(
        semantic_module, "_embed_text", lambda text: np.zeros(4, dtype=np.float32)
    )

    response = await semantic_module.handle_reindex({"clear_existing": False})
    payload = json.loads(response[0].text)
    assert payload["status"] == "error"
    assert "clear_existing=true" in payload["message"]


def test_upsert_refuses_dim_mismatch_leaves_index_unchanged(
    semantic_test_env, monkeypatch
):
    """_upsert_index_record refuses a write whose dimension differs from the
    existing index (B23): returns False and leaves both the rows and index_meta
    untouched — it must NOT insert a mixed-dim row or overwrite meta to the new
    model. This is the single authoritative write path, so this refusal is what
    makes mixed-dim indexes structurally impossible."""
    _index_three_papers()  # dim-3 rows + meta (ST, dim 3)
    meta_before = semantic_module._read_index_meta()

    # The active model now emits a different dimension than the stored index.
    monkeypatch.setattr(
        semantic_module, "_embed_text", lambda text: np.zeros(4, dtype=np.float32)
    )

    result = semantic_module._upsert_index_record(
        paper_id="2405.00009",
        title="Mismatch",
        abstract="whatever",
        authors=[],
        categories=[],
    )
    assert result is False

    conn = semantic_module._connect()
    try:
        rows = conn.execute(
            "SELECT paper_id, embedding_dim FROM semantic_index"
        ).fetchall()
    finally:
        conn.close()
    # No new row, and no 4-dim row landed.
    assert len(rows) == 3
    assert all(int(r["embedding_dim"]) == 3 for r in rows)
    assert "2405.00009" not in {r["paper_id"] for r in rows}
    # Meta NOT overwritten to the current (mismatched) model.
    assert semantic_module._read_index_meta() == meta_before


@pytest.mark.asyncio
async def test_paper_id_incompatible_index_returns_reindex_guidance(
    semantic_test_env, monkeypatch
):
    """F6 + F3: paper_id mode on a legacy (meta-free) index after a model switch
    returns the actionable reindex guidance via the UP-FRONT identity check —
    NOT the generic 'Could not index source paper' — and performs NO write. The
    identity is inferred as the legacy MiniLM model (F3), and the check fires
    before any index-on-miss attempt (F6)."""
    # Reset the module lock onto this test's loop (a sibling lock-contending test
    # may have bound it to a now-closed loop).
    monkeypatch.setattr(semantic_module, "_reindex_lock", None)

    _index_three_papers()  # dim-3 rows, meta stamped ST/MiniLM
    # Legacy pre-B23 index: strip meta so the identity must be INFERRED (F3).
    conn = semantic_module._connect()
    try:
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()

    # Simulate a model switch: the active model differs from the legacy identity.
    monkeypatch.setattr(
        semantic_module.settings, "EMBEDDING_MODEL", "minishlab/potion-retrieval-32M"
    )

    # The up-front guard (F6) must fire BEFORE any index-on-miss attempt.
    def _boom(pid):
        raise AssertionError("index-on-miss must not run on an incompatible index")

    monkeypatch.setattr(semantic_module, "index_paper_by_id", _boom)

    response = await semantic_module.handle_semantic_search({"paper_id": "9999.00001"})
    assert "reindex" in response[0].text.lower()

    # No write happened; meta still absent; still exactly 3 dim-3 rows.
    conn = semantic_module._connect()
    try:
        rows = conn.execute("SELECT embedding_dim FROM semantic_index").fetchall()
        meta_count = conn.execute("SELECT COUNT(*) AS c FROM index_meta").fetchone()
    finally:
        conn.close()
    assert len(rows) == 3
    assert all(int(r["embedding_dim"]) == 3 for r in rows)
    assert meta_count["c"] == 0


def test_upsert_uses_begin_immediate_transaction(semantic_test_env, monkeypatch):
    """F1: the compatibility read and the write live in ONE explicit write
    transaction opened with BEGIN IMMEDIATE (spanning the SELECT so a second
    process cannot slip a conflicting write in between — TOCTOU)."""
    statements = []
    real_connect = semantic_module._connect

    def _traced_connect():
        conn = real_connect()
        conn.set_trace_callback(lambda s: statements.append(s))
        return conn

    monkeypatch.setattr(semantic_module, "_connect", _traced_connect)

    assert (
        semantic_module._upsert_index_record(
            paper_id="2406.00001",
            title="T",
            abstract="transformer vision",
            authors=[],
            categories=[],
        )
        is True
    )

    upper = [s.strip().upper() for s in statements]
    assert any(s.startswith("BEGIN IMMEDIATE") for s in upper)
    # The write happens INSIDE that transaction: INSERT after BEGIN, then COMMIT.
    begin_idx = next(i for i, s in enumerate(upper) if s.startswith("BEGIN IMMEDIATE"))
    insert_idx = next(
        i for i, s in enumerate(upper) if s.startswith("INSERT INTO SEMANTIC_INDEX")
    )
    assert begin_idx < insert_idx
    assert any(s.startswith("COMMIT") for s in upper)


def test_upsert_refuses_same_dim_model_switch(semantic_test_env, monkeypatch):
    """F2: a model switch at the SAME dimension is refused by the identity check
    (a dimension-only check would not catch it) — no write, meta untouched."""
    _index_three_papers()  # dim-3, meta = ST/MiniLM
    meta_before = semantic_module._read_index_meta()

    # Same backend + dimension (DummyModel stays dim-3), different model name.
    monkeypatch.setattr(
        semantic_module.settings, "EMBEDDING_MODEL", "different/same-dim-model"
    )

    result = semantic_module._upsert_index_record(
        paper_id="2407.00001",
        title="T",
        abstract="transformer vision study",
        authors=[],
        categories=[],
    )
    assert result is False

    conn = semantic_module._connect()
    try:
        rows = conn.execute(
            "SELECT paper_id, embedding_dim FROM semantic_index"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 3
    assert "2407.00001" not in {r["paper_id"] for r in rows}
    assert all(int(r["embedding_dim"]) == 3 for r in rows)
    assert semantic_module._read_index_meta() == meta_before


@pytest.mark.asyncio
async def test_legacy_index_pro_st_matches_and_first_upsert_stamps_meta(
    semantic_test_env, monkeypatch
):
    """F3 (matching direction): a legacy (meta-free) index under the [pro-st]
    defaults (ST/MiniLM == the inferred legacy identity) is compatible — searches
    work unchanged, and the first upsert re-stamps index_meta."""
    monkeypatch.setattr(semantic_module, "_reindex_lock", None)
    _index_three_papers()
    # Strip meta -> legacy pre-B23 index. Active backend is ST/MiniLM (fixture, no
    # EMBEDDING_MODEL override) == the inferred legacy identity.
    conn = semantic_module._connect()
    try:
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()

    # Search works (no reindex error) because identities match.
    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 3}
    )
    payload = json.loads(response[0].text)
    assert payload["total_results"] == 3
    assert payload["papers"][0]["id"] == "2403.00001"

    # Meta stays absent until a write happens...
    assert semantic_module._read_index_meta() == {}
    # ...and the next (same-identity, allowed) upsert stamps it.
    assert (
        semantic_module._upsert_index_record(
            paper_id="2403.00004",
            title="More",
            abstract="transformer vision extra",
            authors=[],
            categories=[],
        )
        is True
    )
    meta = semantic_module._read_index_meta()
    assert (
        meta["embedding_model"]
        == "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2"
    )
    assert meta["embedding_dim"] == "3"


def test_legacy_meta_free_model_switch_refuses_write(semantic_test_env, monkeypatch):
    """F3 (mismatching direction): a legacy (meta-free) index whose INFERRED
    identity (MiniLM) differs from the active model is refused at the write path,
    even at the same dimension. Mutation target for the legacy-identity inference."""
    _index_three_papers()
    conn = semantic_module._connect()
    try:
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()

    # Same dimension (DummyModel dim-3), different model -> identity mismatch vs
    # the inferred legacy identity.
    monkeypatch.setattr(
        semantic_module.settings, "EMBEDDING_MODEL", "different/same-dim-model"
    )

    result = semantic_module._upsert_index_record(
        paper_id="2408.00001",
        title="T",
        abstract="transformer vision",
        authors=[],
        categories=[],
    )
    assert result is False

    conn = semantic_module._connect()
    try:
        rows = conn.execute("SELECT paper_id FROM semantic_index").fetchall()
        meta_count = conn.execute("SELECT COUNT(*) AS c FROM index_meta").fetchone()
    finally:
        conn.close()
    assert len(rows) == 3
    assert "2408.00001" not in {r["paper_id"] for r in rows}
    assert meta_count["c"] == 0  # a refused write never stamps meta


@pytest.mark.asyncio
async def test_legacy_meta_free_model_switch_query_mode_reindex_error(
    semantic_test_env, monkeypatch
):
    """F3: a legacy (meta-free) index + a same-dim model switch trips the reindex
    guard in QUERY mode too (identity inferred as the legacy MiniLM model)."""
    monkeypatch.setattr(semantic_module, "_reindex_lock", None)
    _index_three_papers()
    conn = semantic_module._connect()
    try:
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(
        semantic_module.settings, "EMBEDDING_MODEL", "different/same-dim-model"
    )

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer"}
    )
    assert "reindex" in response[0].text.lower()


@pytest.mark.asyncio
async def test_reindex_no_clear_refuses_model_switch(semantic_test_env, monkeypatch):
    """F2 on the rebuild path: reindex(clear_existing=false) refuses when the
    active model IDENTITY differs from the index's, even at the same dimension."""
    monkeypatch.setattr(semantic_module, "_reindex_lock", None)
    _index_three_papers()  # meta = ST/MiniLM, dim-3

    # Same dimension (DummyModel probe -> dim-3), different model identity.
    monkeypatch.setattr(
        semantic_module.settings, "EMBEDDING_MODEL", "different/same-dim-model"
    )

    response = await semantic_module.handle_reindex({"clear_existing": False})
    payload = json.loads(response[0].text)
    assert payload["status"] == "error"
    assert "clear_existing=true" in payload["message"]


@pytest.mark.asyncio
async def test_reindex_clear_probe_failure_leaves_index_intact(
    semantic_test_env, monkeypatch
):
    """R1: reindex(clear_existing=true) probes the active model BEFORE the DELETE.
    If the model fails to load, the index is left untouched and an error is
    returned — a recoverable index is not destroyed by a typo'd/offline model."""
    monkeypatch.setattr(semantic_module, "_reindex_lock", None)
    _index_three_papers()  # 3 rows written (via DummyModel) BEFORE the probe stub

    def _boom(text):
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(semantic_module, "_embed_text", _boom)

    response = await semantic_module.handle_reindex({"clear_existing": True})
    payload = json.loads(response[0].text)
    assert payload["status"] == "error"
    assert "left untouched" in payload["message"]

    # The DELETE never ran — all rows are intact.
    conn = semantic_module._connect()
    try:
        n = conn.execute("SELECT COUNT(*) AS c FROM semantic_index").fetchone()["c"]
    finally:
        conn.close()
    assert n == 3


@pytest.mark.asyncio
async def test_reindex_clear_all_papers_fail_returns_error(
    semantic_test_env, monkeypatch, temp_storage_path
):
    """R1: after a clear, if there were papers to index and NONE succeeded, the
    result is status 'error' (not 'success' with indexed=0) — the cleared index
    is now empty and that must not read as success."""
    monkeypatch.setattr(semantic_module, "_reindex_lock", None)
    Path(temp_storage_path, "2301.00001.md").write_text("paper", encoding="utf-8")
    Path(temp_storage_path, "2301.00002.md").write_text("paper", encoding="utf-8")

    # Probe succeeds (DummyModel via the fixture), but every per-paper index fails.
    monkeypatch.setattr(semantic_module, "index_paper_by_id", lambda pid: False)

    response = await semantic_module.handle_reindex({"clear_existing": True})
    payload = json.loads(response[0].text)
    assert payload["status"] == "error"
    assert payload["indexed"] == 0
    assert set(payload["failed"]) == {"2301.00001", "2301.00002"}


def test_model2vec_empty_string_zero_vector_no_nan(monkeypatch):
    """model2vec returns a zero vector for empty text; _embed_text leaves it as-is
    (no divide-by-norm -> no NaN), and a zero query vector ranks with score 0."""

    class _ZeroStatic:
        def encode(self, text):
            if not text:
                return np.zeros(3, dtype=np.float32)
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    class _Factory:
        @staticmethod
        def from_pretrained(name, normalize=False, force_download=True):
            return _ZeroStatic()

    monkeypatch.setattr(semantic_module, "SentenceTransformer", None)
    monkeypatch.setattr(semantic_module, "StaticModel", _Factory)
    monkeypatch.setattr(semantic_module.settings, "EMBEDDING_MODEL", None)
    _reset_model_globals(monkeypatch)

    zero = semantic_module._embed_text("")
    assert np.all(zero == 0.0)
    assert not np.any(np.isnan(zero))

    candidates = [
        {
            "paper_id": "p1",
            "title": "t",
            "abstract": "a",
            "authors": [],
            "categories": [],
            "published": "",
            "vector": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
    ]
    ranked = semantic_module._rank_by_similarity(zero, candidates, max_results=3)
    assert len(ranked) == 1
    assert ranked[0].score == 0.0
    assert not np.isnan(ranked[0].score)
