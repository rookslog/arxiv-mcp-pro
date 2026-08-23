[![PyPI](https://img.shields.io/pypi/v/arxiv-mcp-pro)](https://pypi.org/project/arxiv-mcp-pro/)
[![Tests](https://github.com/loganrooks/arxiv-mcp-pro/actions/workflows/tests.yml/badge.svg)](https://github.com/loganrooks/arxiv-mcp-pro/actions/workflows/tests.yml)
[![GitHub Stars](https://img.shields.io/github/stars/loganrooks/arxiv-mcp-pro?style=flat)](https://github.com/loganrooks/arxiv-mcp-pro/stargazers)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Codex Plugin](https://img.shields.io/badge/Codex-Plugin-412991?style=flat-square)](./.codex-plugin/plugin.json)

# ArXiv MCP Pro

<!-- mcp-name: io.github.loganrooks/arxiv-mcp-pro -->

> 🔍 The efficient, reliable, citation-capable arXiv MCP server — a token-frugal, standalone evolution of [`arxiv-mcp-server`](https://github.com/blazickjp/arxiv-mcp-server) by Joseph Blazick (Pearl Labs).

ArXiv MCP Pro provides a bridge between AI assistants and arXiv's research repository through the Model Context Protocol (MCP). It lets AI models search for papers and access their content programmatically — with token-frugal, paginated outputs and citation tooling layered on the original server's foundation.

<div align="center">
  
🤝 **[Contribute](https://github.com/loganrooks/arxiv-mcp-pro/pulls)** • 
📝 **[Report Bug](https://github.com/loganrooks/arxiv-mcp-pro/issues)**
</div>

## ✨ Core Features

- 🔎 **Paper Search**: Query arXiv papers with filters for date ranges and categories
- 📄 **Paper Access**: Download and read paper content
- 📋 **Paper Listing**: View all downloaded papers
- 🗃️ **Local Storage**: Papers are saved locally for faster access
- 📝 **Prompts**: A set of research prompts for paper analysis



## 🔒 Security

### Prompt Injection Risk

**Paper content retrieved from arXiv is untrusted external input.**

When an AI assistant downloads or reads a paper through this server, the paper's
text is passed directly into the model's context. A maliciously crafted paper
could embed adversarial instructions designed to hijack the AI's behavior — for
example, instructing it to exfiltrate data, invoke other tools with unintended
arguments, or override system-level instructions. This is a known class of
attack described by OWASP as **LLM01: Prompt Injection** and by the OWASP
Agentic AI framework as **AG01: Prompt Injection in LLM-Integrated Systems**.

### Recommended Mitigations

1. **Use read-only MCP configurations** — where possible, configure the MCP
   client so that the arxiv-mcp-pro server cannot trigger write operations or invoke
   other tools on your behalf.
2. **Review paper content before acting on AI summaries** — if an AI summary
   asks you to run commands or visit external URLs that were not part of your
   original request, treat that as a red flag.
3. **Be cautious in multi-tool setups** — agentic pipelines that combine this
   server with filesystem, shell, or browser tools are higher risk; a prompt
   injection in a paper could chain tool calls unexpectedly.
4. **Treat AI-generated summaries as data, not instructions** — always apply
   human judgment before executing any action the AI recommends after reading a
   paper.

### References

- [OWASP LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OWASP Agentic AI - AG01: Prompt Injection](https://genai.owasp.org/llmrisk/ag01-prompt-injection/)

---

## 🚀 Quick Start

### Install

`arxiv-mcp-pro` is published on [PyPI](https://pypi.org/project/arxiv-mcp-pro/). Install it,
or run it with no install at all via `uvx`:

```bash
# Install from PyPI
pip install arxiv-mcp-pro

# ...or skip the install and point your MCP client at uvx (see the
# integration examples below — running `uvx arxiv-mcp-pro` by hand starts
# the stdio server, which waits silently for an MCP client):
uvx arxiv-mcp-pro

# Add the [pdf] extra for older, PDF-only papers:
pip install "arxiv-mcp-pro[pdf]"
```

Verify the install (the server has no `--help` yet — it speaks MCP over
stdio, so invoking it bare just waits for a client):

```bash
pip show arxiv-mcp-pro
```

> **PDF fallback (older papers):** Most arXiv papers have an HTML version handled
> automatically. Older papers that only have a PDF need the `[pdf]` extra
> (pymupdf4llm) — install it with `pip install "arxiv-mcp-pro[pdf]"`.

**macOS desktop bundle:** a one-click Claude Desktop `.mcpb` bundle (macOS, Apple Silicon)
is attached to each [GitHub Release](https://github.com/loganrooks/arxiv-mcp-pro/releases).

### Development install

To hack on the server from a local clone:

```bash
# Clone the repository
git clone https://github.com/loganrooks/arxiv-mcp-pro.git
cd arxiv-mcp-pro

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install with the test extra (dev + test dependencies)
uv pip install -e ".[test]"
```

### 🤖 Codex Plugin Integration

This repository now includes a Codex plugin manifest at `.codex-plugin/plugin.json`
and a portable MCP config at `.mcp.json` so Codex-oriented tooling can discover
the server without inventing its own install recipe.

The Codex integration uses the same stdio launch path documented elsewhere in
this README:

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "sh",
      "args": ["scripts/run-server.sh"]
    }
  }
}
```

If your Codex client supports plugin manifests, point it at
`./.codex-plugin/plugin.json`. If it only supports raw MCP configuration, use
`./.mcp.json` directly.

### 🔌 MCP Integration

Point your MCP client at the published package with `uvx` — no separate install
step required:

```json
{
    "mcpServers": {
        "arxiv": {
            "command": "uvx",
            "args": [
                "arxiv-mcp-pro",
                "--storage-path", "/path/to/paper/storage"
            ]
        }
    }
}
```

If you have already installed the package (`pip install arxiv-mcp-pro`), invoke the
console script directly instead:

```json
{
    "mcpServers": {
        "arxiv": {
            "command": "arxiv-mcp-pro",
            "args": ["--storage-path", "/path/to/paper/storage"]
        }
    }
}
```

For development against a local clone, run it in place with `uv --directory`:

```json
{
    "mcpServers": {
        "arxiv": {
            "command": "uv",
            "args": [
                "--directory",
                "/path/to/cloned/arxiv-mcp-pro",
                "run",
                "arxiv-mcp-pro",
                "--storage-path", "/path/to/paper/storage"
            ]
        }
    }
}
```

### HTTP Transport

For server deployments where stdio is not practical, run the server with Streamable HTTP:

```bash
TRANSPORT=http HOST=127.0.0.1 PORT=8080 arxiv-mcp-pro --storage-path /path/to/papers
```

Then configure an MCP client that supports Streamable HTTP:

```json
{
    "mcpServers": {
        "arxiv": {
            "type": "http",
            "url": "http://127.0.0.1:8080/mcp"
        }
    }
}
```

The default HTTP bind host is `127.0.0.1`. Streamable HTTP enables MCP DNS rebinding protection by default and allows loopback hosts for the configured port. If exposing the server through a reverse proxy, keep it bound to localhost unless you have added authentication and network controls upstream; set `ALLOWED_HOSTS` and `ALLOWED_ORIGINS` to the external host/origin values your proxy forwards.

## 🔒 Security Note

arXiv papers are user-generated, untrusted content. Paper text returned by this
server may contain prompt injection attempts — crafted text designed to manipulate
an AI assistant's behavior. Treat all paper content as untrusted input.

In production environments, apply appropriate sandboxing and avoid feeding raw
paper content into agentic pipelines that have access to sensitive tools or data
without review. See [SECURITY.md](SECURITY.md) for the full security policy.

## 💡 Available Tools

### Core Workflow

The typical workflow for deep paper research is:

```
search_papers → download_paper → read_paper
```

`list_papers` shows what you have locally. `semantic_search` searches across your local collection.

---

### 1. Paper Search
Search arXiv with optional category, date, and boolean filters. Enforces arXiv's 3-second rate limit automatically — including across parallel sessions on one machine. If rate limited anyway, follow the wait time in the error message before retrying.

```python
result = await call_tool("search_papers", {
    "query": "\"KAN\" OR \"Kolmogorov-Arnold Networks\"",
    "max_results": 10,
    "date_from": "2024-01-01",
    "categories": ["cs.LG", "cs.AI"],
    "sort_by": "date"   # or "relevance" (default)
})
```

Supported categories include `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `cs.NE`, `stat.ML`, `math.OC`, `quant-ph`, `eess.SP`, and more. See tool description for the full list.

The query, category, and date clauses are joined with an explicit `AND`, so `categories` is a **strict** filter (a bare space is not `AND` on the arXiv API — it ranks loosely, closer to `OR`). The response reports `total_results` (the corpus-wide count of papers matching the query, from the feed's `opensearch:totalResults`, falling back to the page size when the feed omits that element) and `returned` (the number of papers in this page). Prefer `sort_by: "relevance"` (the default) for topical search; `sort_by: "date"` returns a recency firehose only weakly filtered by the query.

### 2. Paper Download
Download a paper by its arXiv ID. Tries HTML first, falls back to PDF. Stores the paper locally for `read_paper` and `semantic_search`. The response includes `content_length`, `returned_chars`, `next_start`, and `is_truncated` so clients can safely page through very large papers without mistaking client-side output caps for failed downloads.

```python
result = await call_tool("download_paper", {
    "paper_id": "2401.12345"
})

# For very large papers, request bounded chunks:
result = await call_tool("download_paper", {
    "paper_id": "2401.12345",
    "start": 0,
    "max_chars": 50000
})
```

> For older papers that only have a PDF, install the `[pdf]` extra: `uv pip install -e ".[pdf]"`

### 3. List Papers
List all papers downloaded locally. Returns arXiv IDs only — use `read_paper` to access content.

```python
result = await call_tool("list_papers", {})
```

### 4. Read Paper
Read the full text of a locally downloaded paper in markdown. **Requires `download_paper` to be called first.** Use `start` and `max_chars` with the returned `next_start` value to page through large papers.

```python
result = await call_tool("read_paper", {
    "paper_id": "2401.12345"
})

result = await call_tool("read_paper", {
    "paper_id": "2401.12345",
    "start": 50000,
    "max_chars": 50000
})
```



## 📝 Research Prompts

The server offers specialized prompts to help analyze academic papers:

### Paper Analysis Prompt
A comprehensive workflow for analyzing academic papers that only requires a paper ID:

```python
result = await call_prompt("deep-paper-analysis", {
    "paper_id": "2401.12345"
})
```

This prompt includes:
- Detailed instructions for using available tools (list_papers, download_paper, read_paper, search_papers)
- A systematic workflow for paper analysis
- Comprehensive analysis structure covering:
  - Executive summary
  - Research context
  - Methodology analysis
  - Results evaluation
  - Practical and theoretical implications
- Future research directions
- Broader impacts

### Pro Prompt Pack

- `summarize_paper`: concise structured summary for one paper.
- `compare_papers`: side-by-side technical comparison across paper IDs.
- `literature_review`: thematic synthesis across a topic and optional paper set.

## ⚙️ Configuration

Configure through command-line options and environment variables:

| Setting | Purpose | Default |
|---------|---------|---------|
| `--storage-path` | Paper storage location | `~/.arxiv-mcp-server/papers` |
| `MAX_RESULTS` | Maximum search results | `50` |
| `REQUEST_TIMEOUT` | API timeout in seconds | `60` |
| `CONTENT_DEFAULT_MAX_CHARS` | Default cap on paper characters returned by `read_paper` / `download_paper` when the caller omits `max_chars` (responses include `is_truncated` / `next_start` for paging). An explicit `max_chars` always wins; `0` (or any non-positive value) disables the default cap (full content). | `60000` |
| `TRANSPORT` | Transport type: `stdio`, `http`, or `streamable-http` | `stdio` |
| `HOST` | Host to bind to in HTTP mode | `127.0.0.1` |
| `PORT` | Port to listen on in HTTP mode | `8000` |
| `ALLOWED_HOSTS` | Comma-separated extra allowed Host header values for Streamable HTTP DNS rebinding protection | empty |
| `ALLOWED_ORIGINS` | Comma-separated extra allowed Origin header values for Streamable HTTP DNS rebinding protection | empty |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional Semantic Scholar API key. When set, `citation_graph` sends it as the `x-api-key` header; per [Semantic Scholar's docs](https://www.semanticscholar.org/product/api) this grants a higher authenticated request-rate limit. When unset, requests are unauthenticated (unchanged behavior). | empty |
| `SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL` | Minimum seconds between Semantic Scholar requests. `0` (default) disables pacing. An authenticated API key grants ~1 request/second across all endpoints, so set this to ~`1.1` to pace requests proactively instead of bursting and relying on 429 retry/backoff. | `0` |
| `ARXIV_MIN_REQUEST_INTERVAL` | Minimum seconds between arXiv API requests (arXiv asks for ≥3s per IP, globally). Sessions on one machine coordinate through a lock file in the storage dir, so parallel/multi-agent use stays under the limit. `0` disables all arXiv pacing (including the cross-process lock file). See "Parallel / multi-agent use" below. | `3` |
| `CITATION_MAX_EDGES` | Optional cap on citation/reference edges returned by `citation_graph`'s legacy (non-paginated) path. Unset (default) returns all edges; a `truncated` flag is added when the cap bites. | empty |
| `EMBEDDING_MODEL` | Optional override for the `semantic_search` embedding model. Unset (default) uses the active backend's default: `minishlab/potion-retrieval-32M` for the lightweight model2vec backend (`[pro]`), or `sentence-transformers/all-MiniLM-L6-v2` for the parity backend (`[pro-st]`). Set it to a model2vec/static-model repo id (model2vec backend) or an ST-compatible id (ST backend). Changing the model invalidates the local index — run the `reindex` tool afterwards. | empty |

### Parallel / multi-agent use

arXiv rate-limits per IP (≈1 request / 3s), globally — not per process. When
several sessions run on one machine (e.g. a fleet of agents), they can jointly
exceed that limit and draw sustained HTTP 429 cooldowns (observed ~3 minutes).

To coordinate, all arXiv API request paths — `search_papers`,
`get_abstract`, `watch_topic`/`check_alerts`, `download_paper`, local resource
listing, and the semantic-index metadata fetches (`reindex`, `semantic_search`'s
on-demand source-paper fetch, and background indexing after downloads) — pace
through a shared lock file in the storage directory
(`<storage-path>/arxiv_api.lock`), so **sessions that share a storage dir stay
under the limit** — one request at a time, spaced by
`ARXIV_MIN_REQUEST_INTERVAL`. Set that to `0` to disable pacing, or raise it to be
more conservative. Note: coordination is per storage dir; **multiple machines
behind one IP remain uncoordinated** (each has its own lock file).

## 🧪 Testing

Run the test suite:

```bash
python -m pytest
```

## 🧪 Experimental Features

> **These features are not yet fully tested and may behave unexpectedly. Use with caution.**

The following tools require additional dependencies and are under active development:

```bash
uv pip install -e ".[pro]"
```

### Semantic Search
Semantic similarity search over your **locally downloaded** papers only. Returns empty results if no papers have been downloaded yet. Requires `[pro]` dependencies.

**Embedding backends.** `[pro]` installs a lightweight [model2vec](https://github.com/MinishLab/model2vec) backend — static, torch-free embeddings (~100 MB incl. numpy/tokenizers), default model `minishlab/potion-retrieval-32M` (512-dim, retrieval-tuned). This replaces the previous sentence-transformers/torch stack (~600 MB larger). If you need byte-identical parity with indexes built before this change, install the original backend instead:

```bash
uv pip install -e ".[pro-st]"   # sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~700 MB via torch)
```

Backend auto-detection: **sentence-transformers wins when it is importable**, so an existing `[pro-st]` (or legacy sentence-transformers) install keeps its 384-dim index working unchanged; otherwise model2vec is used. Set `EMBEDDING_MODEL` to override the active backend's default model. Switching backends or models changes the embedding dimension/vectors, so **run the `reindex` tool** afterwards — until you do, `semantic_search` detects the mismatch and returns a clear "run reindex" message rather than mis-ranking or crashing.

```python
result = await call_tool("semantic_search", {
    "query": "test-time adaptation in multimodal transformers",
    "max_results": 5
})
# or find papers similar to a known paper:
result = await call_tool("semantic_search", {
    "paper_id": "2404.19756",
    "max_results": 5
})
```

### Citation Graph
Fetch references and citing papers via Semantic Scholar. Works on any arXiv ID — no local download required.

```python
result = await call_tool("citation_graph", {
    "paper_id": "2401.12345"
})
```

Optional parameters (all opt-in; omit them for the legacy full-output behavior):

- `limit` (integer, 1–1000): max edges per direction, using Semantic Scholar's paginated endpoints.
- `offset` (integer, ≥ 0): pagination offset. Applies **only** together with `limit` or `compact`; passing `offset` alone is ignored and falls through to the legacy path.
- `compact` (boolean): strips author lists + nested `external_ids` and minifies the JSON for lower token cost.
- `counts_only` (boolean): return **only** the paper's true citation/reference totals as `total_citations`/`total_references` (Semantic Scholar's scalar `citationCount`/`referenceCount`) with no edge lists — one endpoint, a small fixed payload. Takes precedence over `limit`/`offset`/`compact`.

> **Note on `citation_count`/`reference_count` in the graph modes:** these count the edges *returned*, not the paper's true totals. The default/legacy nested call is capped by Semantic Scholar at 1000 per direction, and in paginated mode they reflect the current page. For authoritative totals use `counts_only: true` (which returns `total_citations`/`total_references`).

When paginating (`limit` or `compact` set), `citation_count`/`reference_count` report the edges returned in the **current page**, not the paper's totals. The response carries a `pagination` block with an independent cursor per direction — `pagination.citations.next` and `pagination.references.next` (each is the next `offset`, or `null` when that direction is exhausted). `offset` is a single value applied to **both** directions, so to page deeply through one direction, advance `offset` to its `next`; the other direction re-paginates from the same `offset` (references are usually small enough to fit one page).

```python
result = await call_tool("citation_graph", {
    "paper_id": "2401.12345",
    "compact": True,
    "limit": 50
})

# True totals only (no edge lists, small fixed payload):
result = await call_tool("citation_graph", {
    "paper_id": "1706.03762",
    "counts_only": True
})
# -> {"counts_only": true, "total_citations": 180624, "total_references": 41, ...}
```

### Research Alerts
Save topic watches and poll for newly published papers since the last check. Uses the same query syntax as `search_papers`.

```python
# Register a watch (idempotent — calling again updates the existing watch)
await call_tool("watch_topic", {
    "topic": "\"multi-agent reinforcement learning\"",
    "categories": ["cs.AI", "cs.LG"],
    "max_results": 10
})

# Check all watches — returns only papers published since last check
result = await call_tool("check_alerts", {})

# Check a single watch
result = await call_tool("check_alerts", {"topic": "\"multi-agent reinforcement learning\""})
```

### Advanced Prompts
`summarize_paper`, `compare_papers`, and `literature_review` for deeper research workflows. Requires `[pro]` dependencies.

---

## 📄 License

Released under the Apache License 2.0. See the LICENSE file for details.

---

<div align="center">

**arxiv-mcp-pro** is maintained by [Logan Rooks](https://github.com/loganrooks) and builds on
[`arxiv-mcp-server`](https://github.com/blazickjp/arxiv-mcp-server), originally created by
Joseph Blazick and the Pearl Labs Team. Released under Apache-2.0.
</div>
