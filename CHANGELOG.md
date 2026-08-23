# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`arxiv-mcp-pro` is the standalone, citation-capable evolution of
[`arxiv-mcp-server`](https://github.com/blazickjp/arxiv-mcp-server) by Joseph Blazick
(Pearl Labs). Releases up to and including v0.5.0 were published under the original name.

## [Unreleased]

### Changed

- **Tool input schemas are now generated from pydantic models instead of being
  hand-written.** Each tool had a JSON Schema dict sitting next to a handler
  that read `arguments` by key, with nothing tying the two together. They could
  drift silently, and cross-cutting schema work meant editing every tool by
  hand — `d22255b` added `additionalProperties: false` to nine files, one line
  at a time. The model is now the single definition; `ToolInput` sets
  `extra="forbid"`, so a closed schema is inherited rather than remembered.

  No client-observable change: all eleven advertised schemas and descriptions
  are byte-identical to the ones they replace, locked in by a snapshot captured
  before the refactor (`tests/fixtures/tool_schema_snapshot.json`). The one
  normalisation is that four tools which omitted `required` now emit
  `required: []` like the other seven, which is semantically the same.

### Fixed

- **CI and fresh installs were broken by `mcp` 2.0.** The dependency was
  specified as `mcp>=1.27.0` with no upper bound. `mcp` 2.0 removed the
  low-level `Server` decorator API — `list_prompts`, `get_prompt`, `list_tools`,
  `call_tool` — that `server.py` is built on, so any environment resolving 2.0
  failed at import with `AttributeError: 'Server' object has no attribute
  'list_prompts'`. Every test module failed at collection, and a fresh
  `pip install arxiv-mcp-pro` produced a server that could not start. Pinned to
  `mcp>=1.27.0,<2` pending a deliberate 2.0 migration.

- **stdio transport: stray writes to stdout no longer corrupt the JSON-RPC
  channel and kill the server.** Under the stdio transport, fd 1 *is* the
  protocol channel. PyMuPDF (pulled in by the `pdf` extra via `pymupdf4llm`)
  binds `sys.stdout` at import time and prints MuPDF diagnostics through it, so
  a single malformed-PDF warning injected non-JSON bytes mid-stream. The client
  failed to decode, the session desynchronised, and the server died of the
  resulting `BrokenPipeError`. Observed in the field three times against an
  OpenAI Secure MCP Tunnel deployment: the tunnel logged `invalid character '='
  looking for beginning of value`, then `stdio MCP command exited`, then served
  HTTP 502 on every subsequent `initialize` — for 16 days, because the tunnel
  process itself stayed up and its health endpoints kept returning 200.

  `_run_stdio` now calls `stdio_guard.protected_stdout()` first, which duplicates
  fd 1 to a private descriptor for the transport and points fd 1 at stderr. The
  redirect is at the descriptor level, so it also covers C-extension writes
  (MuPDF is a C library) and references to stdout captured before startup —
  neither of which a `sys.stdout` reassignment would intercept. Diagnostics are
  redirected, not discarded: they land on stderr.

## [0.8.0] - 2026-07-17

Repo polish after the v0.7.0 PyPI release, plus reliability/ergonomics fixes
driven by multi-agent field use. Headlines: `search_papers`' category filter is
now a strict `AND` (it was silently advisory), query text is properly
percent-encoded, `total_results` is the real corpus-wide count, and every
arXiv call path paces through a cross-process rate limiter so parallel agent
fleets sharing one machine stay under arXiv's per-IP limit.

### Added
- Contributor scaffolding: `CONTRIBUTING.md`, GitHub issue templates (bug / feature),
  a pull request template, and a Dependabot config (weekly `pip` + `github-actions`
  updates). Also a committed `.claude/settings.json` with a minimal permissions allowlist.
- **Cross-process arXiv rate pacing** (B17): the arXiv-API request paths now pace through a
  lock file in the storage dir (`arxiv_api.lock`), so multiple sessions on one machine that
  share a storage dir stay under arXiv's ≈1-request/3s per-IP limit — the failure mode where
  a fleet of parallel agents drew sustained HTTP 429 cooldowns. Paced paths: `search_papers`
  (both the arxiv-library and raw-HTTP/date routes), `get_abstract`, `watch_topic` /
  `check_alerts` (via `_raw_arxiv_search`), and `download_paper`'s PDF-fallback metadata
  fetch. A 429/503 publishes a shared cooldown (`arxiv_api.cooldown`) so sibling lanes back
  off together rather than each rediscovering the limit. New `ARXIV_MIN_REQUEST_INTERVAL`
  knob (default `3`s; `0` disables all pacing including the lock/cooldown files). Fail-open:
  any pacer error degrades to in-process pacing and never breaks a request. See the README
  "Parallel / multi-agent use" note; multiple machines behind one IP remain uncoordinated.
  The remaining call sites — the semantic-index metadata fetches and local resource
  listing — are paced too (see the B20 Fixed entry below).

### Changed
- **`search_papers` reports the real corpus-wide match count** (B16). `total_results`
  is now the arXiv feed's `opensearch:totalResults` — the total number of papers matching
  the query — instead of `len(page)` (which was always ≤ `max_results` and misleading). A
  new `returned` field carries the page size (the count `total_results` used to report).
  `total_results` falls back to the page size on the rare occasion the feed omits the
  `opensearch:totalResults` element. `search_papers` also now runs through a **single code
  path**: both date-filtered and plain queries route through the raw-HTTP helper (the
  arxiv-package branch, which space-joined its clauses, is gone), so behaviour no longer
  diverges by whether a date filter is present. Two consequences of retiring the
  arxiv-package branch for previously-package-path (non-date) queries: the `published`
  field is now serialized as the feed's raw timestamp (e.g. `2023-01-01T00:00:00Z`)
  rather than a Python `isoformat()` string (`...+00:00`); and error messages use the
  raw-path wording — the old `Error: ArXiv API error - ...` prefix is gone, and rate-limit
  and HTTP errors now surface the raw path's messages.
- **`read_paper` / `download_paper` now cap the default content return** at
  `CONTENT_DEFAULT_MAX_CHARS` (60000 chars) when `max_chars` is omitted (B12).
  Uncapped whole-paper defaults (~137k chars observed) overflowed MCP clients'
  per-tool-output limits and blocked the read path entirely. Responses have
  always carried `is_truncated` / `next_start`, so capped reads are pageable;
  an explicit `max_chars` still wins, and setting the env var to `0` (or any
  non-positive value) restores the legacy full-content default. Note: a
  schema-violating explicit `max_chars` (`0`, negative — the schema minimum
  is 1) is ignored and now receives the default cap, where it previously
  received full content.
- CI tuning: added `concurrency` (auto-cancel superseded runs) to the `CI`, `Lint`,
  and `Run Tests` workflows, and pip dependency caching to `CI` and `Lint` (the `Run
  Tests` matrix installs via `uv`, so pip caching does not apply — a uv cache is a
  follow-up); removed the dead `master` branch triggers and leftover template comments.
- **Slimmer install**: dropped `aiohttp`, `python-dotenv`, `anyio`, and `sse-starlette`
  from the runtime dependencies (none were imported anywhere; `anyio`/`sse-starlette`
  still arrive via `mcp` and dotenv support via `pydantic-settings`, so behavior is
  unchanged) and pinned `arxiv>=2.1,<3` — arxiv 4.x pulls in lxml (~20 MB compiled)
  and had never been what the lockfile and test suite ran against. A clean
  `pip install` shrinks from ~77 MB / 46 packages to ~53 MB / 40 packages.

### Fixed
- **`search_papers`' category filter is now a strict `AND`** (B16). Non-date queries
  previously went through the arxiv Python package, which joined the query group and the
  category group with a bare space — and a space is *not* `AND` on the arXiv API (it ranks
  loosely, closer to `OR`), so the `categories` filter was advisory rather than strict and
  off-topic results leaked in (e.g. `survey` + `cs.HC OR cs.CY` returned astronomy
  sky-surveys). All queries now route through the raw-HTTP helper, which joins clauses with
  an explicit `AND`. Every outbound GET inside the rate-limited helper now also records the
  request against the cross-process pacer clock (previously only the pre-request pace was
  recorded).
- **`search_papers` query text is now properly percent-encoded** (B16). The old
  encoder only turned spaces into `+`, so reserved characters in a query silently corrupted
  the request: `C#` truncated the URL at the `#` (dropping the category/date/`max_results`
  parameters that followed), `R&D` split the query at the `&`, and `C++` reached arXiv as
  two spaces. User-supplied query and category text is now percent-encoded (`urllib.parse.quote`),
  keeping only the arXiv query syntax literal (quotes for phrases, parentheses for grouping,
  the colon in field prefixes like `ti:`/`cat:`); the boolean operators and the date filter's
  literal `+TO+` are unaffected. `categories` values are additionally validated against a
  strict token grammar, so a value carrying whitespace, a boolean operator, a wildcard, or a
  URL delimiter (e.g. `cs.AI OR all:*`, `cs.AI&max_results=1000`) is rejected rather than
  interpolated into the URL; a well-formed but unknown subcategory (`cs.NOTREAL`) is still
  accepted as before.
- **`search_papers` no longer mangles old-style arXiv ids with a `v` in the name**
  (B16). The short-id parser stripped from the first `v` rather than a terminal version
  suffix, so `solv-int/9501001v1` collapsed to `sol`; it now strips only a trailing `vN`
  (`solv-int/9501001v1` → `solv-int/9501001`, `2401.12345v2` → `2401.12345`).
- **`search_papers` / `check_alerts` now surface arXiv API errors instead of a bogus
  result** (B16). arXiv reports a bad query as an HTTP-200 Atom feed with a single
  `/api/errors` entry (and `opensearch:totalResults` of 1), which the unified raw-HTTP
  parser treated as a real 1-paper result; it now detects that entry and raises with the
  feed's error text, so `search_papers` returns an `Error:` message and `check_alerts`
  records it as that topic's per-topic error (restoring the behaviour the deleted
  arxiv-package path had).
- **Saved watch-topic categories can no longer bypass category validation** (B16).
  `watch_topic` now validates `categories` at save time (rejecting a malformed value before
  it is persisted), and `_raw_arxiv_search` enforces the strict category-token grammar as a
  backstop, so a malformed/injection value in a stored watch (e.g. `cs.AI OR all:*`) is
  rejected before any request is built rather than interpolated into the arXiv URL.
- **The remaining arXiv call sites now pace through the cross-process limiter** (B20). The
  semantic-index metadata fetches — `PaperManager.store_paper` / `list_resources` and
  `index_paper_by_id` (used by the `reindex` loop, `semantic_search`'s on-demand source-paper
  fetch, and `download_paper`'s background re-index) — previously bypassed the pacer with a
  fresh, unpaced `arxiv.Client()` and could burst past the ≈1-request/3s limit under
  parallel/multi-agent use. They now route through the shared pacer via a new synchronous
  entry point (`pace_arxiv_request_sync`) for the worker-thread callers, so a fleet sharing a
  storage dir stays under the limit on every arXiv path.
- `reindex` and `semantic_search`'s missing-source indexing now run off the event loop
  (`asyncio.to_thread`), so their newly-paced, one-request-per-paper loops no longer freeze
  every other tool; and `semantic_search` now serializes behind a running `reindex` (a shared
  in-process lock) instead of reading a just-cleared index mid-rebuild (B20).
- `search_papers` now documents which timestamp `date_from`/`date_to` bind to: arXiv's
  `submittedDate`, the original (v1) submission time — which can differ from the arXiv-ID
  prefix month and the latest-version date on cross-listed/revised papers, so strict
  windows should be widened slightly and hits verified against their `published` field
  (B18).
- README no longer claims a PyPI package is "planned for a future release" — the package
  shipped to PyPI on 2026-07-05. Install docs now lead with `pip install arxiv-mcp-pro` /
  `uvx arxiv-mcp-pro`, a PyPI version badge is added, the macOS `.mcpb` desktop bundle on
  GitHub Releases is referenced, and the stale "until the PyPI package ships" MCP-config
  note is gone.
- Agent docs corrected: `CLAUDE.md` mis-expanded MCP as "Message Control Protocol"
  (→ *Model Context Protocol*), listed only 4 of the ~10 tool modules, and documented
  non-existent `ARXIV_*`-prefixed env vars; `AGENTS.md` (previously stale instructions for a
  removed tool) is now a lean, tracked agent guide.
- `semantic_search`'s missing-dependency hint (tool description + runtime error) now
  gives the published-package command (`pip install "arxiv-mcp-pro[pro]"`) instead of
  the source-checkout-only `uv pip install -e ".[pro]"`, which fails for pip/uvx
  installs (B15).
- `search_papers`' non-date (arxiv-library) path no longer free-rides outside the rate
  pacer — it previously read the pacer clock but never acquired the lock or updated the
  timestamp, so those searches were effectively unpaced (B17).
- arXiv rate-limit errors now respect a short `Retry-After` header (≤30s → one retry) and
  otherwise fail fast with an honest, actionable message (naming the server's requested
  delay, or noting observed cooldowns can reach ~3 minutes under parallel use) instead of
  the previous hardcoded "wait 60 seconds" (B17).
- The Semantic Scholar request pacer now clamps the computed wait itself, not just the
  configured interval — float round-up in the schedule arithmetic could push a single
  pacing sleep a few ULPs past the 30s bound, which surfaced as a timing-dependent CI
  test flake (`assert 30.00000000000003 <= 30.0`) and technically violated the pacer's
  bounded-sleep guarantee (B21).

## [0.7.0] - 2026-06-27

First release published to PyPI under the **`arxiv-mcp-pro`** name (via PyPI
Trusted Publishing / OIDC). `pip install arxiv-mcp-pro`.

### Added
- **`library_influence`** — a descriptive influence panel over your *downloaded*
  library (C5). Builds the induced citation subgraph across local papers (no
  database), then ranks them by personalised PageRank alongside global/local
  citation counts, author pedigree (max co-author h-index), and a has-code
  signal. The load-bearing column is `local_vs_global_delta` — where a paper's
  standing *inside* your corpus disagrees with its global citation footprint
  (a corpus "hidden gem"). Reuses `citation_graph`'s Semantic Scholar pacing /
  backoff / optional API key, fetching references and counts via the
  `/paper/batch` endpoint (chunked at 100 ids). Descriptive only — the
  predictive layer stays gated on a pre-registered backtest. Requires the
  opt-in `[influence]` extra (`pip install "arxiv-mcp-pro[influence]"`,
  pulls `networkx`+`scipy`); the base install is unaffected, and the tool
  degrades gracefully with an install hint when the extra is absent.

### Changed
- **Release pipeline wired** — `publish.yml` now uses PyPI Trusted Publishing
  (OIDC) instead of a stored API token; `lint.yml` is check-only (`black
  --check`) and no longer auto-commits formatting back to a protected branch.
- Trimmed the published **sdist** to source + essential metadata (was sweeping in
  repo/agent internals like `CLAUDE.md`, `.claude/`, `.github/`); the wheel was
  already scoped to the package.

### Removed
- Dropped `black` from the **runtime** dependencies (it is a dev tool, used only
  via pre-commit / the `dev` extra — it was never imported at runtime).

## [0.6.0] - 2026-06-26

First release under the **`arxiv-mcp-pro`** name — the project detached from the upstream
fork network and rebranded. Theme: token-frugal citation tooling and reliability hygiene on
an enforced-by-construction CI/review foundation.

### Added
- **`citation_graph` uplift** — opt-in `limit`/`offset`/`compact` pagination and a
  `counts_only` mode that returns true Semantic Scholar scalar totals
  (`total_citations`/`total_references`) at near-zero token cost; optional
  `SEMANTIC_SCHOLAR_API_KEY` (x-api-key) and a configurable request-pacing interval.
- **`semantic_search` uplift** — opt-in `offset`/`compact` pagination (default output is
  byte-for-byte unchanged).
- Paginated paper-content responses (`start`/`max_chars`) for `download_paper` /
  `read_paper`.

### Changed
- **Rebranded to `arxiv-mcp-pro`** — package name, primary console script `arxiv-mcp-pro`
  (with an `arxiv-mcp-server` back-compat alias), branding, funding and security metadata.
  Joseph Blazick retained as original author (Apache-2.0 lineage). The `arxiv_mcp_server`
  import package name is unchanged.
- `citation_graph` now retries with backoff on HTTP 429 and caps returned edges; in the
  graph modes `citation_count`/`reference_count` count the edges returned (use `counts_only`
  for true totals).

### Fixed
- **`check_alerts` per-topic resilience** — one topic's transient failure no longer aborts
  the whole batch; `last_checked` advances per successful topic via an incremental atomic
  save, and a failed topic carries an `error` field and retries next run (B6).
- **`watched_topics.json` atomic write** (`temp` + `fsync` + `os.replace`) — an interrupted
  save can no longer truncate the file and drop all watches (B5).
- `citation_graph` API-key hygiene — key restricted to printable ASCII and sanitized before
  use (no-leak).
- `get_abstract` — dropped a dead `_last_request_time` import (B9).
- arXiv PDF downloads are streamed via `httpx` to avoid truncated files.

### Infrastructure
- Enforcement-by-construction gates: `black`, secret-scan, and citation / claim-ledger lints
  run via pre-commit and GitHub Actions, with a cross-platform test matrix
  (Python 3.11 / 3.12 × ubuntu / macOS / windows).
- Tiered code-review protocol plus a branch-protection-enforced cross-vendor (Codex) merge
  gate for higher-risk changes.

> Distribution note: v0.6.0 is tagged from source. A published PyPI package
> (`uvx arxiv-mcp-pro`) and a one-click Claude Desktop `.mcpb` bundle are planned for a
> future release; both publish workflows fire on a published GitHub Release.

## [0.5.0] - 2026-05-18

_Published as `arxiv-mcp-server`._

### Added
- Streamable HTTP transport (#94).
- Claude Desktop MCPB packaging (#87).
- Tool annotations (#102); trusted-publishing-to-PyPI CI.

### Fixed
- MCP error flag set on tool error payloads (#95).
- Closed MCP tool schemas and aligned server metadata (#104).
- Alert response-shape test coverage (#100).

## [0.4.12] and earlier

Published as `arxiv-mcp-server` by Joseph Blazick. See the
[commit history](https://github.com/loganrooks/arxiv-mcp-pro/commits/main) for details.

[Unreleased]: https://github.com/loganrooks/arxiv-mcp-pro/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/loganrooks/arxiv-mcp-pro/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/loganrooks/arxiv-mcp-pro/releases/tag/v0.7.0
[0.6.0]: https://github.com/loganrooks/arxiv-mcp-pro/releases/tag/v0.6.0
[0.5.0]: https://github.com/loganrooks/arxiv-mcp-pro/releases/tag/v0.5.0
