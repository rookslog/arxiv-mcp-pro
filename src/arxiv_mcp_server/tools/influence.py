"""Local-library influence panel (C5).

Builds the induced citation subgraph over the papers you have downloaded
locally and shows a side-by-side panel of influence signals. The product
insight is the *disagreement* between columns: a paper with high local
PageRank but low global citationCount is a corpus-specific hidden gem.

Design notes
------------
* Nodes are the locally-downloaded arXiv IDs (PaperManager.list_papers()),
  canonicalized to their unversioned form so they match Semantic Scholar's
  (unversioned) identifiers and reference ids.
* Edges are *induced*: A -> B when A references B AND both A and B are in the
  local library (no 1-hop expansion to non-downloaded references in v1).
* The primary rank is a personalised PageRank over that induced subgraph
  (networkx). The personalization vector is uniform over the (optionally
  restricted) library, so it reduces to standard PageRank today; it is passed
  explicitly so a future seeded variant is a one-line change and so the
  degradation paths are unambiguous.

Offline-testability is a hard requirement: all network and filesystem I/O is
isolated behind injectable helpers (``_fetch_s2_batch``, the PaperManager) and
the ranking math lives in the pure ``build_influence_rows`` function, which is
tested directly with hand-built fixtures and zero network access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

import httpx
from pydantic import Field
import mcp.types as types
from ..schemas import ToolInput, schema_from_model
from mcp.types import ToolAnnotations

# Reuse (do not duplicate) the Semantic Scholar plumbing from citation_graph.
from .citation_graph import (
    SEMANTIC_SCHOLAR_BASE_URL,
    RETRYABLE_STATUS,
    _auth_headers,
    _backoff_delay,
    _pace_request,
)
from .list_papers import is_valid_arxiv_id

try:  # pragma: no cover - exercised via the runtime check / monkeypatched flag
    import networkx as nx
except ImportError:  # pragma: no cover - handled gracefully in the runtime check
    nx = None  # type: ignore[assignment]

try:  # pragma: no cover - probed only; networkx.pagerank needs scipy at runtime
    import scipy as _scipy
except ImportError:  # pragma: no cover - handled gracefully in the runtime check
    _scipy = None  # type: ignore[assignment]

logger = logging.getLogger("arxiv-mcp-pro")

# Semantic Scholar's batch endpoint accepts up to 500 ids per request, but it is
# ALSO bounded by response size — and we request `references.externalIds`, whose
# nested lists inflate the payload, so a 500-id chunk can exceed that size limit
# and fail the whole panel. Chunk conservatively at 100 (a handful of extra 1-RPS
# calls for a large library, vs an all-or-nothing failure). Flagged independently
# by the Opus + Codex reviews; the exact safe ceiling is a live-verification item.
S2_BATCH_MAX_IDS = 100

# Fields requested from the batch endpoint. `references.externalIds` carries
# each reference's identifiers (we look for the `ArXiv` external id) to build
# induced edges; `authors.hIndex` powers the author-pedigree column.
S2_BATCH_FIELDS = (
    "title,year,citationCount,influentialCitationCount,"
    "authors.hIndex,references.externalIds"
)

# Case-insensitive markers that indicate code is available for a paper. Matched
# against the paper's local markdown (substring, lower-cased).
HAS_CODE_PATTERNS = (
    "github.com",
    "gitlab.com",
    "paperswithcode.com",
    "code is available",
    "code available at",
    "our code",
    "implementation is available",
)


class LibraryInfluenceInput(ToolInput):
    """Arguments for the `library_influence` tool."""

    compact: Optional[bool] = Field(
        default=None,
        description=(
            "Return minified JSON (lower token cost), mirroring citation_graph's compact convention."
        ),
    )
    paper_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict the panel to this subset of arXiv IDs. Omit to use the entire local library."
        ),
    )
    top_k: int = Field(
        default=20,
        ge=1,
        description=(
            "Maximum number of rows to return, ranked by local PageRank descending (default: 20)."
        ),
    )


library_influence_tool = types.Tool(
    name="library_influence",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Descriptive influence panel over your locally-downloaded library. "
        "Builds the induced citation subgraph (A->B when downloaded paper A "
        "references downloaded paper B) and returns a side-by-side panel of "
        "signals per paper: personalised local PageRank, local in-library "
        "citation count, Semantic Scholar global citationCount / "
        "influentialCitationCount, the max author h-index, whether code appears "
        "available, and a `local_vs_global_delta` disagreement signal "
        "(positive = ranks higher locally than globally = a corpus-specific "
        "hidden gem). Operates on the whole local library by default; restrict "
        "with `paper_ids`. Requires the 'influence' extra: "
        "pip install arxiv-mcp-pro[influence]."
    ),
    inputSchema=schema_from_model(LibraryInfluenceInput),
)


def _dependency_error_envelope() -> List[types.TextContent]:
    """Return the JSON error envelope used when the `influence` extra is absent."""
    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "status": "error",
                    "message": (
                        "library_influence requires the 'influence' extra: "
                        "pip install arxiv-mcp-pro[influence]"
                    ),
                }
            ),
        )
    ]


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    """Yield successive `size`-length chunks of `items`."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


_ARXIV_VERSION_RE = re.compile(r"v\d+$")


def _normalize_arxiv_id(raw: str) -> str:
    """Strip a trailing arXiv version suffix (vN) to the canonical id.

    `2401.12345v2` -> `2401.12345`; `hep-th/9901001v3` -> `hep-th/9901001`;
    already-bare ids are returned unchanged. Local download stems may be
    versioned (download_paper stores under whatever id was passed), but
    Semantic Scholar's `externalIds.ArXiv` and the batch lookup are
    UNVERSIONED. So all id matching — the S2 query, induced edges, the
    paper_ids filter, and the output arxiv_id — must happen on the canonical
    id; the original (possibly versioned) stem is used only to read the local
    markdown file."""
    return _ARXIV_VERSION_RE.sub("", raw)


def _text_has_code(text: str) -> bool:
    """True when the markdown text contains any code-availability marker."""
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in HAS_CODE_PATTERNS)


def _extract_arxiv_references(s2_paper: Dict[str, Any]) -> List[str]:
    """Pull the canonical ArXiv ids of a paper's references from an S2 entry.

    Each reference is an S2 paper stub; we read its `externalIds.ArXiv` (the
    batch endpoint returns `references.externalIds`). References without an
    ArXiv id are skipped (they cannot be local-library nodes in v1). The id is
    normalized to its unversioned form so BOTH sides of the induced-edge join
    are canonical — S2 may return a versioned reference id (e.g. `2401.0002v1`)
    that must still match the canonical library node `2401.0002`."""
    references = s2_paper.get("references") or []
    arxiv_ids: List[str] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        external_ids = reference.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv") if isinstance(external_ids, dict) else None
        if arxiv_id:
            arxiv_ids.append(_normalize_arxiv_id(str(arxiv_id)))
    return arxiv_ids


def _max_author_hindex(s2_paper: Dict[str, Any]) -> Optional[int]:
    """Max hIndex over the paper's S2 authors, or None if unknown."""
    authors = s2_paper.get("authors") or []
    hindices = [
        author.get("hIndex")
        for author in authors
        if isinstance(author, dict) and isinstance(author.get("hIndex"), (int, float))
    ]
    return max(hindices) if hindices else None


def _compute_pagerank(graph: "nx.DiGraph") -> Dict[str, float]:
    """Personalised PageRank with graceful degradation to uniform.

    Degradation contract:
      * 0 nodes -> {} (no rows).
      * nodes but 0 edges -> uniform 1/N (every node is dangling; returned
        directly so the result is uniform regardless of solver availability).
      * a non-converging power iteration (PowerIterationFailedConvergence) ->
        uniform 1/N rather than a raised exception.
    The personalization vector is uniform over the nodes, so this reduces to
    standard PageRank but is explicit about intent and degradation."""
    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return {}
    uniform = {node: 1.0 / n for node in nodes}
    if graph.number_of_edges() == 0:
        return uniform
    try:
        return nx.pagerank(graph, personalization=dict(uniform))
    except nx.PowerIterationFailedConvergence:  # pragma: no cover - defensive
        logger.warning("PageRank failed to converge; degrading to uniform ranks")
        return uniform


def _ordinal_ranks(library_ids: List[str], key) -> Dict[str, int]:
    """1-based ordinal ranks (rank 1 = best) by `key` descending.

    Ties are broken by arXiv id (ascending) so ranks are total and
    deterministic, which keeps `local_vs_global_delta` reproducible."""
    ordered = sorted(library_ids, key=lambda aid: (-key(aid), aid))
    return {aid: position + 1 for position, aid in enumerate(ordered)}


def build_influence_rows(
    library_ids: List[str],
    s2_data: Dict[str, Optional[Dict[str, Any]]],
    has_code_map: Dict[str, bool],
    top_k: int,
) -> Dict[str, Any]:
    """Pure core: build the induced graph and rank the library.

    Inputs are already-fetched (no I/O here):
      * ``library_ids`` — the node set (arXiv IDs).
      * ``s2_data`` — arXiv id -> raw S2 batch entry (or None if S2 had no
        record for that id).
      * ``has_code_map`` — arXiv id -> bool.
      * ``top_k`` — max rows returned (ranked by pagerank desc).

    `local_vs_global_delta` is the disagreement signal. Both `pagerank_rank`
    and `global_citations_rank` are 1-based ordinal ranks (rank 1 = highest
    value). The delta is::

        local_vs_global_delta = global_citations_rank - pagerank_rank

    A paper that is rank 2 locally but rank 40 globally gets +38: it punches
    above its global weight *within your library* (a hidden gem). Negative =
    globally celebrated but locally peripheral. Papers with a null global
    citationCount (e.g. absent from S2) get a null `local_vs_global_delta`:
    the disagreement is undefined without a global citation count, so it is
    NOT fabricated as an inflated positive (which would let an absent-from-S2
    paper masquerade as a hidden gem). Such papers still participate in the
    PageRank ranking and still appear as rows. Likewise the delta is null when
    `local_citations` is 0: a zero-in-degree paper is tied at the PageRank
    dangling floor, so its ordinal rank would be decided alphabetically and the
    delta there is noise, not signal. The raw `pagerank` floor value is still
    reported for those rows.

    NOTE on the spec wording: the design states the delta as "rank-by-pagerank
    position MINUS rank-by-global-citations position". That equals the formula
    above only when "position" is read as ascending (position 1 = lowest). To
    guarantee the stated semantics (positive = hidden gem) under the universal
    "rank 1 = best" convention, the implementation computes
    `global_rank - pagerank_rank` (algebraically identical to the spec formula
    under the ascending-position reading); see the DEVIATION note in the build
    report."""
    graph = nx.DiGraph()
    graph.add_nodes_from(library_ids)
    library_set = set(library_ids)

    for arxiv_id in library_ids:
        s2_paper = s2_data.get(arxiv_id)
        if not s2_paper:
            continue
        for reference_id in _extract_arxiv_references(s2_paper):
            if reference_id in library_set and reference_id != arxiv_id:
                graph.add_edge(arxiv_id, reference_id)

    # Round PageRank ONCE up front and use the same rounded value everywhere —
    # the ordinal ranking that feeds the delta, the emitted `pagerank`, and the
    # row sort — so the displayed order can never disagree with the delta's
    # implied pagerank_rank within 1e-6.
    raw_pagerank = _compute_pagerank(graph)
    pagerank = {aid: round(float(raw_pagerank.get(aid, 0.0)), 6) for aid in library_ids}

    pagerank_rank = _ordinal_ranks(library_ids, key=lambda aid: pagerank[aid])

    def _global_citation_value(arxiv_id: str) -> float:
        entry = s2_data.get(arxiv_id) or {}
        value = entry.get("citationCount")
        # None/unknown sort last (treated as least-cited) deterministically.
        return float(value) if isinstance(value, (int, float)) else -1.0

    global_rank = _ordinal_ranks(library_ids, key=_global_citation_value)

    rows: List[Dict[str, Any]] = []
    zero_indegree_nulled = False
    for arxiv_id in library_ids:
        s2_paper = s2_data.get(arxiv_id) or {}
        global_citations = s2_paper.get("citationCount")
        has_global = isinstance(global_citations, (int, float))
        local_citations = int(graph.in_degree(arxiv_id))

        # The delta is only meaningful for a paper with BOTH a global count and
        # at least one in-library citation. A null global count makes the
        # disagreement undefined (don't fabricate a hidden gem). A zero-in-degree
        # paper sits at the PageRank dangling floor, tied with every other
        # zero-in-degree paper, so its ordinal rank — hence the delta — is
        # decided alphabetically by arXiv id: noise, not signal. Null it; the
        # raw `pagerank` floor value is still reported as an honest reading.
        if has_global and local_citations > 0:
            delta: Optional[int] = global_rank[arxiv_id] - pagerank_rank[arxiv_id]
        else:
            delta = None
            if local_citations == 0:
                zero_indegree_nulled = True

        rows.append(
            {
                "arxiv_id": arxiv_id,
                "title": s2_paper.get("title") or "",
                "pagerank": pagerank[arxiv_id],
                "local_citations": local_citations,
                "global_citations": global_citations,
                "influential_citations": s2_paper.get("influentialCitationCount"),
                "max_author_hindex": _max_author_hindex(s2_paper),
                "has_code": bool(has_code_map.get(arxiv_id, False)),
                "local_vs_global_delta": delta,
            }
        )

    # Sort by pagerank desc, tie-break by arXiv id for determinism, then top_k.
    rows.sort(key=lambda row: (-row["pagerank"], row["arxiv_id"]))
    rows = rows[: max(1, top_k)] if rows else rows

    notes: List[str] = []
    missing = [aid for aid in library_ids if not s2_data.get(aid)]
    if missing:
        notes.append(
            f"{len(missing)} paper(s) were not found on Semantic Scholar; their "
            "global signals are null, so their local_vs_global_delta is null "
            "(the disagreement is undefined without a global citation count)."
        )
    if any(s2_data.get(aid) for aid in library_ids):
        notes.append(
            "Induced citation edges are best-effort: the Semantic Scholar batch "
            "endpoint may cap the number of references returned per paper, so "
            "some edges among your library may be missing."
        )
    if zero_indegree_nulled:
        notes.append(
            "local_vs_global_delta is only meaningful for papers cited at least "
            "once within your library; papers with local_citations == 0 have a "
            "null delta."
        )

    return {
        "library_size": len(library_ids),
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "papers": rows,
        "notes": notes,
    }


async def _s2_post(
    client: httpx.AsyncClient,
    url: str,
    json_body: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> httpx.Response:
    """POST with the same pacing/backoff discipline as citation_graph._s2_get.

    Retries on RETRYABLE_STATUS responses and httpx.TransportError up to
    max_retries times, honoring a numeric Retry-After (clamped) and otherwise
    using jittered exponential backoff. Returns the final response (the caller
    still calls raise_for_status())."""
    response: Optional[httpx.Response] = None
    for attempt in range(max_retries + 1):
        await _pace_request()
        try:
            response = await client.post(url, json=json_body, headers=headers or {})
        except httpx.TransportError:
            if attempt == max_retries:
                raise
            await asyncio.sleep(_backoff_delay(None, base_delay, attempt))
            continue
        if response.status_code not in RETRYABLE_STATUS or attempt == max_retries:
            return response
        retry_after = response.headers.get("Retry-After")
        await asyncio.sleep(_backoff_delay(retry_after, base_delay, attempt))
    return response


async def _fetch_s2_batch(
    arxiv_ids: List[str],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Fetch S2 signals for every library paper via the batch endpoint.

    Chunked to S2_BATCH_MAX_IDS per request. The batch endpoint returns a list
    aligned by index with the input ids, with a null entry for any id S2 has no
    record of. Returns arXiv id -> S2 entry (or None). This is the injectable
    network seam: tests monkeypatch this function so no real HTTP happens.

    The timeout is 60s (vs citation_graph's 30s single-paper calls): one batch
    call fans the WHOLE library plus nested `references.externalIds`, so the
    payload can be large. The exact response size — and whether S2 caps
    references per paper in batch mode — is a live-verification item."""
    result: Dict[str, Optional[Dict[str, Any]]] = {aid: None for aid in arxiv_ids}
    if not arxiv_ids:
        return result

    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/batch?fields={S2_BATCH_FIELDS}"
    headers = _auth_headers()
    async with httpx.AsyncClient(timeout=60.0) as client:
        for chunk in _chunked(arxiv_ids, S2_BATCH_MAX_IDS):
            body = {"ids": [f"ARXIV:{aid}" for aid in chunk]}
            response = await _s2_post(client, url, body, headers=headers)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                # Defensive: a non-list payload (e.g. an error dict) must NOT be
                # zipped over — that would iterate dict KEYS and silently map
                # papers by key order. Treat it as "no records this chunk".
                logger.warning(
                    "S2 batch returned a non-list payload; treating as empty"
                )
                payload = []
            for arxiv_id, entry in zip(chunk, payload):
                # entry is None when S2 has no record for that id.
                result[arxiv_id] = entry if isinstance(entry, dict) else None
    return result


def _get_paper_manager() -> Any:
    """Construct a PaperManager.

    Imported lazily so importing this module never requires the optional `pdf`
    extra (resources.papers imports pymupdf4llm at module load, but the panel
    only needs list_papers / get_paper_content). Also the test seam: tests
    monkeypatch this factory to inject a stub manager (no filesystem)."""
    from ..resources.papers import PaperManager

    return PaperManager()


async def _build_has_code_map(
    arxiv_ids: List[str],
    manager: Any,
    id_to_stem: Optional[Dict[str, str]] = None,
) -> Dict[str, bool]:
    """Scan each paper's local markdown for code-availability markers.

    Keys the result by the (canonical) `arxiv_ids` but reads the file by the
    original local stem via `id_to_stem` (canonical -> stem); when no map is
    given the id is used as the stem directly. Resilient per-paper: a paper
    whose markdown cannot be read scans as False rather than failing the whole
    panel."""
    id_to_stem = id_to_stem or {}
    has_code: Dict[str, bool] = {}
    for arxiv_id in arxiv_ids:
        stem = id_to_stem.get(arxiv_id, arxiv_id)
        try:
            content = await manager.get_paper_content(stem)
        except Exception as exc:  # noqa: BLE001 - per-paper resilience
            logger.debug("has_code scan skipped for %s: %s", stem, exc)
            content = ""
        has_code[arxiv_id] = _text_has_code(content)
    return has_code


async def handle_library_influence(
    arguments: Dict[str, Any],
) -> List[types.TextContent]:
    """Wire PaperManager + S2 batch fetch + markdown scan into the pure core."""
    try:
        # networkx.pagerank dispatches to its scipy solver (no pure-python
        # fallback in networkx>=3), so BOTH must be present. Gating on both
        # yields the helpful `[influence]` install hint instead of a raw
        # "No module named 'scipy'" surfacing through the broad except below.
        if nx is None or _scipy is None:
            return _dependency_error_envelope()

        # Defense-in-depth coercion (schema enforces minimum 1 / array / bool).
        raw_top_k = arguments.get("top_k", 20)
        top_k = max(1, int(raw_top_k)) if raw_top_k is not None else 20
        compact = arguments.get("compact") is True
        requested_ids = arguments.get("paper_ids")

        manager = _get_paper_manager()
        stems = await manager.list_papers()
        # Keep only valid arXiv-ID stems — a stray non-paper `.md` in the storage
        # dir would otherwise become a bogus `ARXIV:<stem>` S2 query + node.
        # Mirrors the public list_papers / semantic-index paths.
        stems = [s for s in stems if is_valid_arxiv_id(s)]

        # The paper_ids filter matches on canonical (unversioned) ids, so a user
        # passing `2401.12345` matches a local stem `2401.12345v2`. Use
        # `is not None` (not truthiness): an explicit `paper_ids: []` restricts
        # to nothing (an empty panel), whereas an omitted paper_ids means
        # "whole library".
        if requested_ids is not None:
            requested = {_normalize_arxiv_id(str(pid)) for pid in requested_ids}
            stems = [s for s in stems if _normalize_arxiv_id(s) in requested]

        # Canonicalize the node set: map each unversioned id to its (first,
        # sorted) local stem. The canonical id is used everywhere (S2 query,
        # induced edges, has_code keys, output arxiv_id); the stem is used only
        # to read the local markdown file.
        canon_to_stem: Dict[str, str] = {}
        collapsed = 0
        for stem in sorted(set(stems)):
            canon = _normalize_arxiv_id(stem)
            if canon in canon_to_stem:
                collapsed += 1
                continue
            canon_to_stem[canon] = stem
        library_ids = sorted(canon_to_stem)

        s2_data = await _fetch_s2_batch(library_ids)
        has_code_map = await _build_has_code_map(
            library_ids, manager, id_to_stem=canon_to_stem
        )

        payload = build_influence_rows(library_ids, s2_data, has_code_map, top_k)
        if collapsed:
            payload["notes"].append(
                f"{collapsed} duplicate-version download(s) were collapsed to a "
                "single canonical arXiv id (the lexicographically first stem was "
                "used to read local content)."
            )
        result = {"status": "success", **payload}

        if compact:
            text = json.dumps(result, separators=(",", ":"))
        else:
            text = json.dumps(result, indent=2)
        return [types.TextContent(type="text", text=text)]
    except Exception as exc:  # noqa: BLE001 - surface a uniform error envelope
        logger.error("library_influence error: %s", exc)
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(exc)}),
            )
        ]
