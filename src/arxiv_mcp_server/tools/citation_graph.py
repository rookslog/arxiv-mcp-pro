"""Citation graph tool using Semantic Scholar API."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
from pydantic import Field
import mcp.types as types
from mcp.types import ToolAnnotations

from ..schemas import ToolInput, schema_from_model
from ..config import Settings

logger = logging.getLogger("arxiv-mcp-pro")
settings = Settings()


def _auth_headers() -> Dict[str, str]:
    """x-api-key header when SEMANTIC_SCHOLAR_API_KEY is configured, else {}.

    Absent key -> no header -> identical unauthenticated behavior. The key is
    stripped and accepted only if it is entirely printable ASCII (0x20-0x7e);
    any other key (control chars, or non-ASCII like U+2028) is dropped and
    warned WITHOUT echoing its value. This makes the no-leak guarantee
    self-contained: a malformed key never reaches the HTTP layer, so it can
    never be echoed back through an h11/httpx exception message into logs or
    returned error text (independent of the dependency's error formatting)."""
    key = (settings.SEMANTIC_SCHOLAR_API_KEY or "").strip()
    if not key:
        return {}
    if any(not (32 <= ord(ch) <= 126) for ch in key):
        logger.warning(
            "Ignoring invalid SEMANTIC_SCHOLAR_API_KEY "
            "(must be printable ASCII; non-printable or non-ASCII characters)"
        )
        return {}
    return {"x-api-key": key}


SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"

# Cap on any single backoff sleep. asyncio.sleep is NOT covered by the httpx
# request timeout, so an unbounded server-supplied Retry-After (or an unbounded
# exponential backoff) could otherwise hang the tool for hours.
MAX_RETRY_DELAY = 16.0

# Transient HTTP statuses worth retrying. 500 is deliberately excluded: it is
# often a permanent server-side error, not a transient one.
RETRYABLE_STATUS = {429, 502, 503, 504}


def _backoff_delay(retry_after, base_delay, attempt):
    """Compute the next sleep, clamped to MAX_RETRY_DELAY.

    A numeric Retry-After (status responses only) is honored but clamped. Absent
    that, full jitter over the clamped exponential window is used."""
    if retry_after is not None and str(retry_after).isdigit():
        return min(float(retry_after), MAX_RETRY_DELAY)
    return random.uniform(0, min(base_delay * (2**attempt), MAX_RETRY_DELAY))


# --- Request pacing (B11) ---------------------------------------------------
# An authenticated Semantic Scholar key grants ~1 request/second across all
# endpoints, so the paginated path's 3 sequential calls otherwise burst past the
# limit and lean on retry/backoff. When SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL is
# > 0, space every S2 request by at least that many seconds. The default (0.0)
# disables pacing, keeping behavior — and timing — exactly as before.

# Upper bound on a single pacing wait, mirroring MAX_RETRY_DELAY: asyncio.sleep
# is not covered by the httpx timeout, so a fat-fingered interval (e.g. 3600, or
# a non-finite value) must not be able to stall the tool — and, since the pace
# lock is held across the sleep, stall every concurrent caller behind it.
MAX_PACE_INTERVAL = 30.0

_pace_lock: Optional[asyncio.Lock] = None
_pace_loop: Optional[asyncio.AbstractEventLoop] = None
_next_request_time = 0.0  # time.monotonic() of the earliest next allowed request


def _get_pace_lock() -> asyncio.Lock:
    """Return a pace lock bound to the *current* running loop.

    A module-global asyncio.Lock binds to the first loop that awaits it; reused
    from a different loop (a second asyncio.run / a per-test loop) it raises
    "bound to a different event loop". Create it lazily per running loop instead.
    Safe without its own guard: there is no await between the check and the
    assignment, so this runs atomically within a single loop."""
    global _pace_lock, _pace_loop
    loop = asyncio.get_running_loop()
    if _pace_lock is None or _pace_loop is not loop:
        _pace_lock = asyncio.Lock()
        _pace_loop = loop
    return _pace_lock


async def _pace_request() -> None:
    """Throttle S2 requests to >= SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL apart.

    No-op when the interval is <= 0 (the default) or non-finite. Otherwise the
    interval is clamped to MAX_PACE_INTERVAL and pacing is serialized via a
    per-loop lock so concurrent callers queue in order rather than all firing at
    once. The clamp guarantees a single pacing wait cannot hang the tool (or,
    since the lock is held across it, every concurrent caller)."""
    interval = settings.SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL or 0.0
    if not math.isfinite(interval) or interval <= 0:
        return
    interval = min(interval, MAX_PACE_INTERVAL)
    global _next_request_time
    async with _get_pace_lock():
        now = time.monotonic()
        # Clamp the wait itself, not just the interval: `now + interval` can
        # round up in float arithmetic, so a back-to-back call may compute a
        # wait a few ULPs above MAX_PACE_INTERVAL (observed in CI:
        # 30.00000000000003). Should any future writer ever move
        # _next_request_time far ahead, this bounds the sleep — and the
        # schedule is then rewritten to now + interval below, not preserved.
        wait = min(_next_request_time - now, MAX_PACE_INTERVAL)
        if wait > 0:
            await asyncio.sleep(wait)
            now = time.monotonic()
        _next_request_time = now + interval


async def _s2_get(client, url, *, headers=None, max_retries=4, base_delay=1.0):
    """GET with backoff on transient failures (S2 rate limits / 5xx / transport).

    Retries on RETRYABLE_STATUS responses and httpx.TransportError up to
    max_retries times (max_retries + 1 total GETs). Honors a numeric Retry-After
    header on status responses (clamped to MAX_RETRY_DELAY); transport errors use
    jittered exponential backoff. Returns the final response (caller still calls
    raise_for_status()).

    Worst-case blocking is bounded: each attempt waits at most MAX_PACE_INTERVAL
    (pacing) plus MAX_RETRY_DELAY (backoff), so a call is bounded by
    ~max_retries * (MAX_PACE_INTERVAL + MAX_RETRY_DELAY); the paginated path makes
    three such calls sequentially. Both sleeps are clamped — neither the pacing
    interval nor a server-supplied Retry-After can hang the tool."""
    response = None
    for attempt in range(max_retries + 1):
        await _pace_request()
        try:
            response = await client.get(url, headers=headers or {})
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


def _apply_edge_cap(
    citations: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Truncate edge lists to settings.CITATION_MAX_EDGES (per direction) when
    configured. Returns (citations, references, truncated: bool).

    A negative cap is treated as "no cap" (graceful handling of bad config) so a
    negative value never produces the surprising `lst[:-n]` slice. cap == 0 is a
    valid "return zero edges" request."""
    cap: Optional[int] = settings.CITATION_MAX_EDGES
    if cap is None or cap < 0:
        return citations, references, False
    truncated = len(citations) > cap or len(references) > cap
    return citations[:cap], references[:cap], truncated


class CitationGraphInput(ToolInput):
    """Arguments for the `citation_graph` tool."""

    compact: Optional[bool] = Field(
        default=None,
        description=(
            "Drop author lists and nested external_ids, return minified id+title edges (lower token cost)."
        ),
    )
    counts_only: Optional[bool] = Field(
        default=None,
        description=(
            "Return ONLY the paper's true citation/reference totals as `total_citations`/`total_references` (Semantic Scholar scalar counts) with no edge lists — one endpoint, a small fixed payload. Takes precedence over limit/offset/compact."
        ),
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description=(
            "Max edges per direction (opt-in pagination; uses Semantic Scholar's paginated endpoints). Omit for legacy full output."
        ),
    )
    offset: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Pagination offset (applies only together with `limit` or `compact`)."
        ),
    )
    paper_id: str = Field(description=("arXiv ID (for example: 2401.12345)."))


citation_graph_tool = types.Tool(
    name="citation_graph",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Return papers citing an arXiv paper and papers that it references "
        "using Semantic Scholar's citation graph. NOTE: in the graph modes "
        "`citation_count`/`reference_count` count the edges RETURNED, not the "
        "paper's true totals — the default/legacy nested call is capped by "
        "Semantic Scholar at 1000 per direction, and in paginated mode they "
        "reflect the current page. For the paper's authoritative totals set "
        "`counts_only: true`: it returns `total_citations`/`total_references` "
        "(Semantic Scholar's scalar citationCount/referenceCount) with no edge "
        "lists — one endpoint, a small fixed payload. In paginated mode "
        "(`limit` or `compact` set) each direction has its own cursor "
        "(`pagination.citations.next` / `pagination.references.next`) to pass as "
        "the next `offset`. If an output cap is configured (CITATION_MAX_EDGES), "
        "a `truncated` flag is added when results were capped (legacy path only)."
    ),
    inputSchema=schema_from_model(CitationGraphInput),
)


def _normalize_paper_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize paper lists returned by Semantic Scholar."""
    normalized: List[Dict[str, Any]] = []
    for item in items:
        paper_id = item.get("paperId")
        title = item.get("title", "")
        year = item.get("year")
        external_ids = item.get("externalIds") or {}
        authors = [author.get("name", "") for author in item.get("authors", [])]

        normalized.append(
            {
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "authors": authors,
                "external_ids": external_ids,
                "arxiv_id": external_ids.get("ArXiv"),
            }
        )

    return normalized


def _normalize_paper_items_compact(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize paper lists into compact edges (no authors/external_ids)."""
    normalized: List[Dict[str, Any]] = []
    for item in items:
        external_ids = item.get("externalIds") or {}
        normalized.append(
            {
                "paper_id": item.get("paperId"),
                "arxiv_id": external_ids.get("ArXiv"),
                "title": item.get("title", ""),
                "year": item.get("year"),
            }
        )
    return normalized


async def _handle_citation_graph_counts(
    paper_id: str,
) -> List[types.TextContent]:
    """Opt-in counts-only mode: return Semantic Scholar's authoritative scalar
    citationCount/referenceCount (the paper's TRUE totals) — one endpoint (no
    citation/reference fan-out), a small fixed payload with no edge lists.

    To avoid one key carrying two meanings, the totals are reported under
    distinct keys `total_citations`/`total_references` — NOT the graph modes'
    `citation_count`/`reference_count`, which count the edges *returned* (S2 caps
    the nested call at 1000 per direction; the paginated path reports the current
    page). The `paper` block mirrors the minimal compact shape (no authors /
    external_ids)."""
    s2_paper_identifier = quote(f"ARXIV:{paper_id}", safe="")
    url = (
        f"{SEMANTIC_SCHOLAR_BASE_URL}/{s2_paper_identifier}"
        "?fields=title,year,citationCount,referenceCount"
    )

    headers = _auth_headers()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await _s2_get(client, url, headers=headers)
        response.raise_for_status()

    payload = response.json()
    result = {
        "status": "success",
        "paper": {
            "paper_id": payload.get("paperId"),
            "arxiv_id": paper_id,
            "title": payload.get("title", ""),
            "year": payload.get("year"),
        },
        "counts_only": True,
        "total_citations": payload.get("citationCount"),
        "total_references": payload.get("referenceCount"),
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_citation_graph_paginated(
    paper_id: str,
    page_limit: int,
    page_offset: int,
    compact: bool,
) -> List[types.TextContent]:
    """Opt-in paginated/compact path using Semantic Scholar's dedicated endpoints."""
    # Defense-in-depth: the schema bounds (1..1000, >=0) are enforced only by the
    # MCP SDK validator. Coerce + clamp per-param here so the handler never trusts
    # the input blindly. NOTE: no `limit + offset <= 1000` sum-clamp — that claim
    # was tested against the live Semantic Scholar API and refuted.
    page_limit = max(1, min(1000, int(page_limit)))
    page_offset = max(0, int(page_offset))

    s2_paper_identifier = quote(f"ARXIV:{paper_id}", safe="")

    # Three sequential requests: root paper metadata, then the /citations and
    # /references pages.
    root_url = (
        f"{SEMANTIC_SCHOLAR_BASE_URL}/{s2_paper_identifier}"
        "?fields=title,year,authors,externalIds"
    )
    page_fields = "title,year,authors,externalIds"
    citations_url = (
        f"{SEMANTIC_SCHOLAR_BASE_URL}/{s2_paper_identifier}/citations"
        f"?fields={page_fields}&limit={page_limit}&offset={page_offset}"
    )
    references_url = (
        f"{SEMANTIC_SCHOLAR_BASE_URL}/{s2_paper_identifier}/references"
        f"?fields={page_fields}&limit={page_limit}&offset={page_offset}"
    )

    headers = _auth_headers()
    async with httpx.AsyncClient(timeout=30.0) as client:
        root_response = await _s2_get(client, root_url, headers=headers)
        root_response.raise_for_status()
        citations_response = await _s2_get(client, citations_url, headers=headers)
        citations_response.raise_for_status()
        references_response = await _s2_get(client, references_url, headers=headers)
        references_response.raise_for_status()

    root_payload = root_response.json()
    citations_payload = citations_response.json()
    references_payload = references_response.json()

    citation_items = [
        entry.get("citingPaper", {}) for entry in citations_payload.get("data", [])
    ]
    reference_items = [
        entry.get("citedPaper", {}) for entry in references_payload.get("data", [])
    ]

    if compact:
        citations = _normalize_paper_items_compact(citation_items)
        references = _normalize_paper_items_compact(reference_items)
        # arxiv_id echoes the input id on every path (legacy + both paginated
        # branches) for a consistent paper.arxiv_id contract.
        paper = {
            "paper_id": root_payload.get("paperId"),
            "arxiv_id": paper_id,
            "title": root_payload.get("title", ""),
            "year": root_payload.get("year"),
        }
    else:
        citations = _normalize_paper_items(citation_items)
        references = _normalize_paper_items(reference_items)
        paper = {
            "paper_id": root_payload.get("paperId"),
            "arxiv_id": paper_id,
            "title": root_payload.get("title", ""),
            "year": root_payload.get("year"),
            "authors": [
                author.get("name", "") for author in root_payload.get("authors", [])
            ],
            "external_ids": root_payload.get("externalIds", {}),
        }

    # NOTE: no _apply_edge_cap here. The caller-supplied `limit` already bounds
    # this path coherently with the per-direction pagination cursors; truncating
    # edges while `pagination.<dir>.next` still reflects S2's uncapped cursor
    # would make a paging client silently skip edges. The output cap is applied
    # ONLY in the legacy/unbounded path.
    result = {
        "status": "success",
        "paper": paper,
        "citation_count": len(citations),
        "reference_count": len(references),
    }
    result.update(
        {
            "citations": citations,
            "references": references,
            "pagination": {
                "limit": page_limit,
                "citations": {
                    "offset": page_offset,
                    "next": citations_payload.get("next"),
                    "returned": len(citations),
                },
                "references": {
                    "offset": page_offset,
                    "next": references_payload.get("next"),
                    "returned": len(references),
                },
            },
        }
    )

    if compact:
        text = json.dumps(result, separators=(",", ":"))
    else:
        text = json.dumps(result, indent=2)

    return [types.TextContent(type="text", text=text)]


async def handle_citation_graph(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle citation graph lookup for a single arXiv paper ID."""
    try:
        paper_id = arguments["paper_id"].strip()
        if not paper_id:
            return [types.TextContent(type="text", text="Error: paper_id is required")]

        limit = arguments.get("limit")
        offset = arguments.get("offset")
        # Strict bool: only a JSON `true` enables compact. Guards against a
        # truthy non-bool (e.g. the string "false") from a non-validating client.
        compact = arguments.get("compact") is True
        # Strict bool, like compact. counts_only takes precedence over the graph
        # modes: it returns the paper's true scalar totals (no edge lists).
        counts_only = arguments.get("counts_only") is True
        if counts_only:
            return await _handle_citation_graph_counts(paper_id)

        # Pagination is triggered only by `limit` or `compact`. `offset` alone is
        # a no-op here (it falls through to the legacy path, which has no paging);
        # it is honored only as a modifier when already paginating.
        if limit is not None or compact:
            page_limit = limit if limit is not None else 100
            page_offset = offset or 0
            return await _handle_citation_graph_paginated(
                paper_id, page_limit, page_offset, compact
            )

        s2_paper_identifier = quote(f"ARXIV:{paper_id}", safe="")
        fields = (
            "title,year,authors,externalIds,"
            "citations.paperId,citations.title,citations.year,citations.authors,citations.externalIds,"
            "references.paperId,references.title,references.year,references.authors,references.externalIds"
        )

        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/{s2_paper_identifier}?fields={fields}"

        headers = _auth_headers()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await _s2_get(client, url, headers=headers)
            response.raise_for_status()

        payload = response.json()
        citations = _normalize_paper_items(payload.get("citations", []))
        references = _normalize_paper_items(payload.get("references", []))

        citations, references, truncated = _apply_edge_cap(citations, references)

        result = {
            "status": "success",
            "paper": {
                "paper_id": payload.get("paperId"),
                "arxiv_id": paper_id,
                "title": payload.get("title", ""),
                "year": payload.get("year"),
                "authors": [
                    author.get("name", "") for author in payload.get("authors", [])
                ],
                "external_ids": payload.get("externalIds", {}),
            },
            "citation_count": len(citations),
            "reference_count": len(references),
        }
        if truncated:
            result["truncated"] = True
        result["citations"] = citations
        result["references"] = references

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except httpx.HTTPStatusError as exc:
        logger.error("Semantic Scholar HTTP error: %s", exc)
        return [
            types.TextContent(
                type="text",
                text=f"Error: Semantic Scholar API HTTP error - {str(exc)}",
            )
        ]
    except Exception as exc:
        logger.error("Citation graph error: %s", exc)
        return [types.TextContent(type="text", text=f"Error: {str(exc)}")]
