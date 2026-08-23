"""Tests for PaperManager arXiv pacing wiring (B20).

``store_paper`` and ``list_resources`` each make an arXiv API call that must be
paced against sibling coroutines AND processes (B20). These tests assert the
pacer is awaited and ``record_arxiv_request`` is called — including on the
finally-path when the arXiv fetch raises. Mock-based: no network, no real
PDF/markdown I/O.
"""

import sys
from unittest.mock import MagicMock

import pytest

from arxiv_mcp_server.config import Settings
from arxiv_mcp_server.resources import papers as papers_module


@pytest.fixture
def paper_manager_env(temp_storage_path, monkeypatch):
    """Point PaperManager at a temp storage dir.

    STORAGE_PATH is a property resolved in ``PaperManager.__init__`` via
    ``Settings()``; replace it on the class (mirrors the pacing suite's ``paced``
    fixture) so the manager stores into tmp rather than the real storage dir.
    The pacing functions themselves are replaced per-test with recorders, so the
    real cross-process lock file is never touched.
    """
    monkeypatch.setattr(Settings, "STORAGE_PATH", temp_storage_path)
    return temp_storage_path


@pytest.mark.asyncio
async def test_store_paper_paces_and_records_on_fetch_error(
    paper_manager_env, monkeypatch
):
    """store_paper awaits the pacer before the fetch and records after, even when
    the arXiv fetch raises (B20 finally-path)."""
    # pymupdf4llm is lazily imported inside store_paper's try block BEFORE the
    # pace; stub it so the test is independent of the [pdf] extra and does no I/O.
    monkeypatch.setitem(sys.modules, "pymupdf4llm", MagicMock())

    events = []

    async def _pace():
        events.append("pace")

    monkeypatch.setattr(papers_module, "pace_arxiv_request", _pace)
    monkeypatch.setattr(
        papers_module, "record_arxiv_request", lambda: events.append("record")
    )

    pm = papers_module.PaperManager()

    def _boom(search):
        events.append("fetch")
        raise RuntimeError("arxiv down")

    pm.client = MagicMock()
    pm.client.results.side_effect = _boom

    with pytest.raises(ValueError):
        await pm.store_paper("2401.00001", "https://arxiv.org/pdf/2401.00001")

    # Paced before the fetch, recorded on the finally-path despite the failure.
    assert events == ["pace", "fetch", "record"]


@pytest.mark.asyncio
async def test_list_resources_paces_once_per_paper(paper_manager_env, monkeypatch):
    """list_resources paces (and records) exactly once per locally stored paper
    id — one arXiv metadata call per loop iteration (B20)."""
    (paper_manager_env / "2401.00001.md").write_text("a", encoding="utf-8")
    (paper_manager_env / "2401.00002.md").write_text("b", encoding="utf-8")

    pace_calls = []

    async def _pace():
        pace_calls.append(1)

    record = MagicMock()
    monkeypatch.setattr(papers_module, "pace_arxiv_request", _pace)
    monkeypatch.setattr(papers_module, "record_arxiv_request", record)

    pm = papers_module.PaperManager()

    mock_paper = MagicMock()
    mock_paper.title = "Some Title"
    mock_paper.summary = "Some abstract"
    pm.client = MagicMock()
    pm.client.results.return_value = [mock_paper]

    resources = await pm.list_resources()

    assert len(pace_calls) == 2  # one pace per stored paper id
    assert record.call_count == 2
    assert len(resources) == 2


@pytest.mark.asyncio
async def test_old_style_id_round_trips_through_paper_manager(
    paper_manager_env, monkeypatch
):
    """PaperManager must expose an encoded cache file under its original ID."""
    paper_id = "hep-th/9901001"
    content = "legacy paper"
    (paper_manager_env / "hep-th%2F9901001.md").write_text(content, encoding="utf-8")

    async def _pace():
        return None

    monkeypatch.setattr(papers_module, "pace_arxiv_request", _pace)
    monkeypatch.setattr(papers_module, "record_arxiv_request", lambda: None)
    search = MagicMock(return_value=object())
    monkeypatch.setattr(papers_module.arxiv, "Search", search)

    pm = papers_module.PaperManager()
    mock_paper = MagicMock(title="Legacy", summary="Abstract")
    pm.client = MagicMock()
    pm.client.results.return_value = [mock_paper]

    assert await pm.list_papers() == [paper_id]
    assert await pm.has_paper(paper_id) is True
    assert await pm.get_paper_content(paper_id) == content
    assert len(await pm.list_resources()) == 1
    search.assert_called_once_with(id_list=[paper_id])
