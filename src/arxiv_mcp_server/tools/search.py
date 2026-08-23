"""Search functionality for the arXiv MCP server."""

import json
import logging
import re
import httpx
import asyncio
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from dateutil import parser
from pydantic import Field
import mcp.types as types
from mcp.types import ToolAnnotations
from ..schemas import ToolInput, schema_from_model
from ..config import Settings
from .arxiv_pacing import (
    pace_arxiv_request,
    record_arxiv_request,
    record_arxiv_cooldown,
)

logger = logging.getLogger("arxiv-mcp-pro")
settings = Settings()

# arXiv asks for >= 3s between requests. The pacer (cross-process aware) lives in
# arxiv_pacing; this constant is kept importable for back-compat (get_abstract
# imports it) and documents the default source-of-truth.
_MIN_REQUEST_INTERVAL = 3.0  # seconds

# Retry-After values at or below this (seconds) are honoured with a single retry;
# longer cooldowns fail fast rather than block the caller. See _rate_limited_get.
_RETRY_AFTER_MAX_SLEEP = 30.0  # seconds


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse an HTTP ``Retry-After`` header into seconds (>= 0).

    Accepts an integer number of seconds or an HTTP-date; returns None when the
    header is absent or unparseable. Past dates (negative deltas) clamp to 0.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    # Integer-seconds form. OverflowError: an absurdly large numeric header
    # (int() is arbitrary-precision; float() overflows) is a parse failure,
    # not an exception to surface.
    try:
        return max(0.0, float(int(value)))
    except ValueError:
        pass
    except OverflowError:
        return None
    # HTTP-date form.
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())


def _rate_limit_message(status_code: int, retry_after: Optional[float]) -> str:
    """Build an honest, actionable arXiv rate-limit error message."""
    # A present-but-non-positive Retry-After (0, negative, or a past HTTP-date
    # all parse to 0.0) carries no useful "wait N seconds" signal — fall back to
    # the generic wording rather than telling the caller to wait 0s. Cap the
    # displayed value (1 day) so an absurd header can't produce a nonsense number.
    if retry_after is not None and retry_after > 0:
        display = int(min(retry_after, 86400))
        return (
            f"arXiv is rate limiting this IP (HTTP {status_code}). "
            f"Server asks for {display}s before retrying."
        )
    return (
        f"arXiv is rate limiting this IP (HTTP {status_code}). "
        "Wait at least 60s before retrying — observed cooldowns can reach "
        "~3 minutes under parallel use."
    )


# Single-source the version from package metadata (settings.APP_VERSION) so the
# User-Agent can't drift from the released version the way a hardcoded string did.
ARXIV_HEADERS = {
    "User-Agent": (
        f"arxiv-mcp-pro/{settings.APP_VERSION} "
        "(https://github.com/loganrooks/arxiv-mcp-pro; research tool)"
    )
}


async def _rate_limited_get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """Make a GET request respecting arXiv's rate limit policy.

    Paces via the cross-process arXiv pacer (:func:`pace_arxiv_request`) so
    sibling agent sessions on one machine stay under arXiv's per-IP limit. EVERY
    outbound GET is paced — the initial attempt, the timeout retry, and the
    429-retry. On 429/503, honours a short ``Retry-After`` (<= 30s) with a single
    retry; otherwise fails fast with an honest, actionable message (retrying
    while rate-limited only extends the ban). One retry on timeout only.
    """
    for attempt in range(2):  # one retry on timeout only
        # Pace at the TOP of each iteration so both the initial attempt and the
        # timeout retry go through the gate (a raised interval must not be
        # undercut by the fixed 5s timeout backoff).
        await pace_arxiv_request()
        try:
            # Pace-before / record-after (matching download.py + papers.py): the
            # GET hit arXiv whether or not it succeeded, so record it in the
            # finally so sibling lanes pace off this attempt even on timeout.
            try:
                response = await client.get(url, headers=ARXIV_HEADERS)
            finally:
                record_arxiv_request()
            if response.status_code in (429, 503):
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                if retry_after is not None and retry_after <= _RETRY_AFTER_MAX_SLEEP:
                    logger.warning(
                        "arXiv rate limited (%s); Retry-After=%.0fs, retrying once",
                        response.status_code,
                        retry_after,
                    )
                    # Publish the shared back-off BEFORE sleeping so sibling
                    # coroutines/processes stop firing immediately, not only after
                    # they each independently hit their own 429.
                    record_arxiv_cooldown(retry_after)
                    # Sleep just the server's Retry-After; the interval is enforced
                    # by the re-pace below (so no need to floor here — a
                    # Retry-After of 0 still can't fire an immediate retry).
                    await asyncio.sleep(retry_after)
                    # Re-pace the retry GET: enforces the interval AND honors any
                    # cooldown published meanwhile, so N callers that slept the
                    # same header don't retry in a thundering herd.
                    await pace_arxiv_request()
                    # Wrap the retry GET's own timeout so it does NOT bubble to the
                    # outer for-loop handler and fire a THIRD request into the
                    # cooldown window — the single-retry contract stops here.
                    try:
                        retry_response = await client.get(url, headers=ARXIV_HEADERS)
                    except httpx.TimeoutException:
                        raise RuntimeError(
                            "arXiv retry after rate-limit cooldown timed out — "
                            "not retrying further"
                        )
                    finally:
                        # The retry GET hit arXiv too — record it (success or
                        # timeout) so sibling lanes pace off the latest attempt.
                        record_arxiv_request()
                    if retry_response.status_code in (429, 503):
                        retry_after2 = _parse_retry_after(
                            retry_response.headers.get("Retry-After")
                        )
                        logger.warning(
                            "arXiv still rate limited (%s) after retry — failing fast",
                            retry_response.status_code,
                        )
                        record_arxiv_cooldown(
                            retry_after2 if retry_after2 is not None else 60.0
                        )
                        raise RuntimeError(
                            _rate_limit_message(
                                retry_response.status_code, retry_after2
                            )
                        )
                    retry_response.raise_for_status()
                    return retry_response
                logger.warning(
                    "arXiv rate limited (%s) — backing off, not retrying",
                    response.status_code,
                )
                record_arxiv_cooldown(retry_after if retry_after is not None else 60.0)
                raise RuntimeError(
                    _rate_limit_message(response.status_code, retry_after)
                )
            response.raise_for_status()
            return response
        except httpx.TimeoutException:
            if attempt == 0:
                logger.warning("arXiv request timed out, retrying once")
                await asyncio.sleep(5.0)
            else:
                raise

    raise RuntimeError("arXiv request timed out after retry")


# arXiv API endpoint for raw queries (bypasses arxiv package URL encoding issues)
# Use HTTPS to avoid redirect from http -> https
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# XML namespaces used in arXiv Atom feed. ``opensearch`` carries the feed-level
# totalResults (the corpus-wide match count, distinct from the page size).
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Valid arXiv category prefixes for validation
VALID_CATEGORIES = {
    "cs",
    "econ",
    "eess",
    "math",
    "physics",
    "q-bio",
    "q-fin",
    "stat",
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nlin",
    "nucl-ex",
    "nucl-th",
    "quant-ph",
}

# A well-formed arXiv category token: a lowercase archive prefix (letters/hyphens,
# e.g. cs, astro-ph, cond-mat, q-bio) optionally followed by a single dotted
# subcategory (e.g. .AI, .gen-ph). Deliberately rejects whitespace, boolean
# operators, wildcards, and URL delimiters (& = # : *) so a category value cannot
# smuggle extra query syntax or parameters into the request URL.
_CATEGORY_TOKEN = re.compile(r"^[a-z][a-z-]*(\.[A-Za-z-]+)?$")

# Characters kept LITERAL when percent-encoding user-supplied query text: the
# double quote (phrase queries), the colon (field prefixes ti:/au:/abs:/cat:),
# and parentheses (grouping). Everything else — including #, &, %, a literal +,
# and any non-ASCII — is percent-encoded so it lands inside the search_query
# value instead of truncating the URL (#), splitting off a spurious parameter
# (&), or being read as a space (+). See _encode_query_text.
_QUERY_SAFE = '":()'


def _encode_query_text(text: str) -> str:
    """Percent-encode user-supplied arXiv query text for the ``search_query`` param.

    Keeps the arXiv query syntax that must survive on the wire literal (double
    quotes for phrases, the colon in field prefixes like ``ti:``/``cat:``, and
    parentheses for grouping) via :data:`_QUERY_SAFE`; percent-encodes everything
    else — notably ``#``, ``&``, ``%``, a literal ``+``, and non-ASCII — so a
    reserved character in user input cannot corrupt the URL. Spaces become
    ``%20`` (arXiv decodes ``%20`` and ``+`` identically inside the query, and the
    boolean operators AND/OR/ANDNOT survive as tokens either way). The structural
    separators (``+AND+``/``+OR+``) and the internally-built date filter's
    ``+TO+`` are the CALLER's literals, added after this call, so they stay
    literal on the wire.
    """
    return quote(text, safe=_QUERY_SAFE)


async def _raw_arxiv_search(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    categories: Optional[List[str]] = None,
    return_total: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Optional[int]]]:
    """
    Perform arXiv search using raw HTTP requests.

    This bypasses the arxiv Python package to avoid URL encoding issues
    with date filters. The arxiv package encodes '+' as '%2B' which breaks
    the submittedDate:[YYYYMMDD+TO+YYYYMMDD] syntax.

    Query, category, and date clauses are joined with an explicit ``AND`` (a bare
    space is NOT ``AND`` on the arXiv API — it ranks loosely, closer to ``OR``,
    which makes a category clause advisory rather than strict).

    By default returns just the list of paper dicts (back-compat for callers like
    ``alerts.py``). When ``return_total`` is True, returns
    ``(papers, total_results)`` where ``total_results`` is the feed-level
    ``opensearch:totalResults`` — the corpus-wide match count, or ``None`` when the
    feed omits it.
    """
    # Build the search_query clauses. Two parallel lists: ``clauses`` are
    # already-URL-encoded fragments (user text percent-encoded via
    # _encode_query_text; structural syntax kept literal) that get joined with the
    # literal ``+AND+`` separator; ``readable`` mirrors them un-encoded for the
    # debug log only.
    clauses: List[str] = []
    readable: List[str] = []

    if query.strip():
        clauses.append(f"({_encode_query_text(query)})")
        readable.append(f"({query})")

    # Add category filtering. This is the SECURITY backstop for callers that reach
    # the helper directly without going through handle_search's _validate_categories
    # (e.g. check_alerts replaying saved-topic categories): every category must pass
    # the strict token grammar before it is interpolated into the URL, so a value
    # carrying whitespace, a boolean operator, a wildcard, or a URL delimiter is
    # rejected here (raising BEFORE any network call) rather than smuggled onto the
    # wire. handle_search still pre-validates for a friendlier early error + prefix
    # warning; this is the last line of defence. Each value is also percent-encoded
    # (defence in depth), while the ``cat:`` prefix and ``+OR+`` separator stay literal.
    if categories:
        for cat in categories:
            if not _CATEGORY_TOKEN.match(cat):
                raise ValueError(f"Invalid category: {cat!r}")
        cat_group = "+OR+".join(f"cat:{_encode_query_text(cat)}" for cat in categories)
        clauses.append(f"({cat_group})")
        readable.append("(" + " OR ".join(f"cat:{cat}" for cat in categories) + ")")

    # Add date filtering using arXiv API syntax
    if date_from or date_to:
        try:
            if date_from:
                start_date = parser.parse(date_from).strftime("%Y%m%d0000")
            else:
                start_date = "199107010000"  # arXiv started July 1991

            if date_to:
                end_date = parser.parse(date_to).strftime("%Y%m%d2359")
            else:
                end_date = datetime.now().strftime("%Y%m%d2359")

            # This filter is internally built (no user text), so it is appended
            # to the encoded clauses verbatim. CRITICAL: the '+' in '+TO+' must
            # remain literal — httpx preserves it on the wire; percent-encoding it
            # to %2B breaks the range syntax.
            date_filter = f"submittedDate:[{start_date}+TO+{end_date}]"
            clauses.append(date_filter)
            readable.append(date_filter)
            logger.debug(f"Added date filter: {date_filter}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing dates: {e}")
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD format: {e}")

    if not clauses:
        raise ValueError("No search criteria provided")

    # Join the clauses with an explicit AND. A bare space is NOT AND on the arXiv
    # API (it ranks loosely, closer to OR), which is why the category filter must
    # be AND-joined to stay strict.
    encoded_query = "+AND+".join(clauses)
    logger.debug(f"Raw API query: {' AND '.join(readable)}")

    # Map sort parameter to arXiv API values
    sort_map = {
        "relevance": "relevance",
        "date": "submittedDate",
    }
    sort_order = "descending"

    # Non-user params (max_results is an int, sortBy is mapped to a fixed token).
    base_params = f"max_results={max_results}&sortBy={sort_map.get(sort_by, 'relevance')}&sortOrder={sort_order}"

    url = f"{ARXIV_API_URL}?search_query={encoded_query}&{base_params}"
    logger.debug(f"Raw API URL: {url}")

    # Make the request via rate-limited helper
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await _rate_limited_get(client, url)

    # Parse the Atom XML response
    papers, total_results = _parse_arxiv_atom_response(response.text)
    if return_total:
        return papers, total_results
    return papers


def _parse_arxiv_atom_response(
    xml_text: str,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Parse arXiv Atom XML response into ``(paper dicts, total_results)``.

    ``total_results`` is the feed-level ``opensearch:totalResults`` — the
    corpus-wide count of papers matching the query, independent of the page size —
    or ``None`` when the element is absent or unparseable.
    """
    results = []
    total_results: Optional[int] = None

    try:
        root = ET.fromstring(xml_text)

        # Feed-level corpus-wide match count (distinct from the page size).
        total_elem = root.find("opensearch:totalResults", ARXIV_NS)
        if total_elem is not None and total_elem.text:
            try:
                total_results = int(total_elem.text.strip())
            except (ValueError, TypeError):
                total_results = None

        for entry in root.findall("atom:entry", ARXIV_NS):
            # Extract paper ID from the id URL
            id_elem = entry.find("atom:id", ARXIV_NS)
            if id_elem is None or id_elem.text is None:
                continue

            id_text = id_elem.text.strip()

            # arXiv reports API errors as an HTTP-200 Atom feed whose single entry
            # has an id under /api/errors, title "Error", and the error text in
            # <summary> (opensearch:totalResults is 1, so this would otherwise look
            # like a genuine 1-paper result and callers would surface a bogus
            # paper). Detect it by the id PATH — robust to http/https and any
            # fragment — and raise so callers treat it as a failure: handle_search
            # returns it as an "Error: ..." message, check_alerts records it as the
            # topic's per-topic error. The deleted arxiv-package path raised on
            # these; this restores that behaviour after unification.
            if urlparse(id_text).path.rstrip("/").endswith("/api/errors"):
                summary_elem = entry.find("atom:summary", ARXIV_NS)
                detail = (
                    summary_elem.text.strip().replace("\n", " ")
                    if summary_elem is not None and summary_elem.text
                    else "unknown error"
                )
                raise ValueError(f"arXiv API error: {detail}")

            # ID format: http://arxiv.org/abs/XXXX.XXXXX or http://arxiv.org/abs/category/XXXXXXX
            paper_id = id_text.split("/abs/")[-1]
            # Strip only a TERMINAL version suffix (v1, v2, ...). A blunt
            # split("v") corrupts old-style archive ids whose name contains a
            # 'v' (e.g. solv-int/9501001v1 -> "sol"); anchor to the end instead.
            short_id = re.sub(r"v\d+$", "", paper_id)

            # Title
            title_elem = entry.find("atom:title", ARXIV_NS)
            title = (
                title_elem.text.strip().replace("\n", " ")
                if title_elem is not None and title_elem.text
                else ""
            )

            # Authors
            authors = []
            for author in entry.findall("atom:author", ARXIV_NS):
                name_elem = author.find("atom:name", ARXIV_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)

            # Abstract/Summary
            summary_elem = entry.find("atom:summary", ARXIV_NS)
            abstract = "[EXTERNAL CONTENT] " + (
                summary_elem.text.strip().replace("\n", " ")
                if summary_elem is not None and summary_elem.text
                else ""
            )

            # Categories
            categories = []
            for cat in entry.findall("arxiv:primary_category", ARXIV_NS):
                term = cat.get("term")
                if term:
                    categories.append(term)
            for cat in entry.findall("atom:category", ARXIV_NS):
                term = cat.get("term")
                if term and term not in categories:
                    categories.append(term)

            # Published date
            published_elem = entry.find("atom:published", ARXIV_NS)
            published = (
                published_elem.text
                if published_elem is not None and published_elem.text
                else ""
            )

            # PDF URL
            pdf_url = None
            for link in entry.findall("atom:link", ARXIV_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break
            if not pdf_url:
                pdf_url = f"http://arxiv.org/pdf/{paper_id}"

            results.append(
                {
                    "id": short_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "categories": categories,
                    "published": published,
                    "url": pdf_url,
                    "resource_uri": f"arxiv://{short_id}",
                }
            )

    except ET.ParseError as e:
        logger.error(f"Failed to parse arXiv XML response: {e}")
        raise ValueError(f"Failed to parse arXiv API response: {e}")

    return results, total_results


class SearchPapersInput(ToolInput):
    """Arguments for the `search_papers` tool."""

    categories: Optional[List[str]] = Field(
        default=None,
        description=(
            "Strongly recommended: arXiv categories to focus search (e.g., ['cs.AI', 'cs.MA'] for agent research, ['cs.LG'] for ML, ['cs.CL'] for NLP, ['cs.CV'] for vision). Greatly improves relevance."
        ),
    )
    date_from: Optional[str] = Field(
        default=None,
        description=(
            "Start date for papers (YYYY-MM-DD format), inclusive. Binds to the original (v1) submission timestamp — see DATE FILTERING in the tool description. Use to find recent work, e.g., '2023-01-01' for last 2 years."
        ),
    )
    date_to: Optional[str] = Field(
        default=None,
        description=(
            "End date for papers (YYYY-MM-DD format), inclusive. Binds to the original (v1) submission timestamp — see DATE FILTERING in the tool description. Use with date_from to find historical work, e.g., '2020-12-31' for older research."
        ),
    )
    max_results: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of results to return (default: 10, max: 50). Use 15-20 for comprehensive searches."
        ),
    )
    query: str = Field(
        description=(
            'Search query using quoted phrases for exact matches (e.g., \'"machine learning" OR "deep learning"\') or specific technical terms. Avoid overly broad or generic terms.'
        )
    )
    sort_by: Optional[Literal["relevance", "date"]] = Field(
        default=None,
        description=(
            "Sort results by 'relevance' (most relevant first, default) or 'date' (newest first). Use 'relevance' for focused searches, 'date' for recent developments."
        ),
    )


search_tool = types.Tool(
    name="search_papers",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="""Search for papers on arXiv with advanced filtering and query optimization.

QUERY SEMANTICS: A bare space between terms is NOT treated as AND by the arXiv API — it ranks
results loosely (closer to OR), which makes a category filter advisory rather than strict. This
server joins the query, category, and date clauses with an explicit AND, so `categories` acts as
a hard filter. Note that `sort_by: "date"` returns a recency firehose only weakly filtered by the
query (observed in field use: near-identical result sets across different queries), so prefer
`sort_by: "relevance"` (the default) for topical search. In the response, `total_results` is the
corpus-wide count of papers matching the query (from the feed's opensearch:totalResults; it falls
back to the page size on the rare occasion the feed omits that element) and `returned` is the
number of papers actually in this page.

QUERY CONSTRUCTION GUIDELINES:
- Use QUOTED PHRASES for exact matches: "multi-agent systems", "neural networks", "machine learning"
- Combine related concepts with OR: "AI agents" OR "software agents" OR "intelligent agents"  
- Use field-specific searches for precision:
  - ti:"exact title phrase" - search in titles only
  - au:"author name" - search by author
  - abs:"keyword" - search in abstracts only
- Use ANDNOT to exclude unwanted results: "machine learning" ANDNOT "survey"
- For best results, use 2-4 core concepts rather than long keyword lists

ADVANCED SEARCH PATTERNS:
- Field + phrase: ti:"transformer architecture" for papers with exact title phrase
- Multiple fields: au:"Smith" AND ti:"quantum" for author Smith's quantum papers  
- Exclusions: "deep learning" ANDNOT ("survey" OR "review") to exclude survey papers
- Broad + narrow: "artificial intelligence" AND (robotics OR "computer vision")

CATEGORY FILTERING (highly recommended for relevance):
Computer Science:
- cs.AI: Artificial Intelligence
- cs.LG: Machine Learning
- cs.CL: Computation and Language (NLP)
- cs.CV: Computer Vision
- cs.MA: Multi-Agent Systems
- cs.RO: Robotics
- cs.NE: Neural and Evolutionary Computing
- cs.IR: Information Retrieval
- cs.HC: Human-Computer Interaction
- cs.CR: Cryptography and Security
- cs.DB: Databases
Statistics & Math:
- stat.ML: Machine Learning (Statistics)
- stat.AP: Applications
- math.OC: Optimization and Control
- math.ST: Statistics Theory
Physics & Other:
- quant-ph: Quantum Physics
- eess.SP: Signal Processing
- eess.AS: Audio and Speech Processing
- physics.data-an: Data Analysis and Statistics

EXAMPLES OF EFFECTIVE QUERIES:
- ti:"reinforcement learning" with categories: ["cs.LG", "cs.AI"] - for RL papers by title
- au:"Hinton" AND "deep learning" with categories: ["cs.LG"] - for Hinton's deep learning work
- "multi-agent" ANDNOT "survey" with categories: ["cs.MA"] - exclude survey papers
- abs:"transformer" AND ti:"attention" with categories: ["cs.CL"] - attention papers with transformer abstracts

DATE FILTERING: Use YYYY-MM-DD format for historical research:
- date_to: "2015-12-31" - for foundational/classic work (pre-2016)
- date_from: "2020-01-01" - for recent developments (post-2020)
- Both together for specific time periods
- Filters bind to arXiv's submittedDate — the ORIGINAL (v1) submission timestamp. For
  cross-listed or revised papers this can differ from the arXiv-ID prefix month and from
  the latest-version date, so papers near a boundary may appear to leak in or out of the
  window. For strict windows, widen the range slightly and verify each hit's published field.

RESULT QUALITY: Default sort is RELEVANCE (most pertinent results first). Use sort_by: "date" to get newest papers first.
Choose relevance for focused topic searches; choose date for monitoring recent developments.

RATE LIMITING: arXiv enforces a 3-second minimum between requests per IP. This server paces requests
automatically, including across parallel sessions on one machine (shared via the storage directory).
If you still see a rate limit error, follow the wait time in the error message before retrying —
do not call the tool repeatedly in a loop; observed cooldowns can reach ~3 minutes.

TIPS FOR FOUNDATIONAL RESEARCH:
- Use date_to: "2010-12-31" to find classic papers on BDI, SOAR, ACT-R
- Combine with field searches: ti:"BDI" AND abs:"belief desire intention"  
- Try author searches: au:"Rao" AND "BDI" for Anand Rao's foundational BDI work""",
    inputSchema=schema_from_model(SearchPapersInput),
)


def _validate_categories(categories: List[str]) -> bool:
    """Validate that all provided categories are well-formed arXiv categories.

    Two gates. First a strict full-token grammar (:data:`_CATEGORY_TOKEN`) rejects
    anything carrying whitespace, boolean operators, wildcards, or URL delimiters
    (``cs.AI OR all:*``, ``cs.AI&max_results=1000``, ``cs AI``) — closing a query/
    parameter-injection vector, since category values are interpolated into the
    request URL. Then the archive prefix must be a known arXiv archive. A
    well-formed but unknown SUBcategory (e.g. ``cs.NOTREAL``) is allowed: arXiv
    tolerates unknown subcategories, so this stays prefix-level like before.
    """
    for category in categories:
        if not _CATEGORY_TOKEN.match(category):
            logger.warning(f"Malformed category token rejected: {category!r}")
            return False
        prefix = category.split(".")[0]
        if prefix not in VALID_CATEGORIES:
            logger.warning(f"Unknown category prefix: {prefix}")
            return False
    return True


def _optimize_query(query: str) -> str:
    """Minimal query optimization - preserve user intent while fixing obvious issues."""

    # Don't modify queries with existing field specifiers (ti:, au:, abs:, cat:)
    if any(
        field in query
        for field in ["ti:", "au:", "abs:", "cat:", "AND", "OR", "ANDNOT"]
    ):
        logger.debug("Field-specific or boolean query detected - no optimization")
        return query

    # Don't modify queries that are already quoted
    if query.startswith('"') and query.endswith('"'):
        logger.debug("Pre-quoted query detected - no optimization")
        return query

    # For very long queries (>10 terms), suggest user be more specific rather than auto-converting
    terms = query.split()
    if len(terms) > 10:
        logger.warning(
            f"Very long query ({len(terms)} terms) - consider using quotes for phrases or field-specific searches"
        )

    # Only optimization: preserve the original query exactly as intended
    return query


async def handle_search(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle paper search requests via the raw arXiv HTTP API.

    All queries — date-filtered or not — route through a single path
    (:func:`_raw_arxiv_search`). It joins the query, category, and date clauses
    with an explicit ``AND`` (a bare space is NOT ``AND`` on the arXiv API — it
    ranks loosely, closer to ``OR``, which made the category filter advisory rather
    than strict in the previous arxiv-package path), paces every request through
    the cross-process rate limiter, and parses the Atom feed. The response reports
    ``total_results`` (the corpus-wide match count from the feed's
    ``opensearch:totalResults``, falling back to the page size when the feed omits
    it) alongside ``returned`` (this page's size).
    """
    try:
        max_results = min(int(arguments.get("max_results", 10)), settings.MAX_RESULTS)
        base_query = arguments["query"]
        date_from_arg = arguments.get("date_from")
        date_to_arg = arguments.get("date_to")
        categories = arguments.get("categories")
        sort_by_arg = arguments.get("sort_by", "relevance")

        logger.debug(
            f"Starting search with query: '{base_query}', max_results: {max_results}"
        )

        # Validate categories if provided
        if categories and not _validate_categories(categories):
            return [
                types.TextContent(
                    type="text",
                    text="Error: Invalid category provided. Please check arXiv category names.",
                )
            ]

        optimized_query = _optimize_query(base_query) if base_query.strip() else ""
        if optimized_query != base_query:
            logger.debug(f"Optimized query: '{base_query}' -> '{optimized_query}'")

        try:
            papers, total = await _raw_arxiv_search(
                query=optimized_query,
                max_results=max_results,
                sort_by=sort_by_arg,
                date_from=date_from_arg,
                date_to=date_to_arg,
                categories=categories,
                return_total=True,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error: {e}")
            return [
                types.TextContent(
                    type="text", text=f"Error: arXiv API HTTP error - {str(e)}"
                )
            ]
        except ValueError as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

        returned = len(papers)
        # Real corpus-wide match count from the feed; fall back to the page size
        # when the feed omits opensearch:totalResults.
        total_results = total if total is not None else returned
        logger.info(
            f"Search completed: {returned} returned of {total_results} total matches"
        )
        response_data = {
            "total_results": total_results,
            "returned": returned,
            "papers": papers,
        }

        return [
            types.TextContent(type="text", text=json.dumps(response_data, indent=2))
        ]

    except Exception as e:
        logger.error(f"Unexpected search error: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]
