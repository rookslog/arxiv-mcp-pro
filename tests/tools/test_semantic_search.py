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
    Path(temp_storage_path, "hep-th%2F9901001.md").write_text(
        "legacy paper", encoding="utf-8"
    )

    indexed_ids = []

    def _mock_index(paper_id):
        indexed_ids.append(paper_id)
        return True

    monkeypatch.setattr(semantic_module, "index_paper_by_id", _mock_index)

    response = await semantic_module.handle_reindex({"clear_existing": True})

    payload = json.loads(response[0].text)
    assert payload["status"] == "success"
    assert set(indexed_ids) == {
        "2301.00001",
        "2301.00002",
        "hep-th/9901001",
    }


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
