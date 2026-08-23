"""Research alert tools for watched topics."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field
import mcp.types as types
from mcp.types import ToolAnnotations

from dateutil import parser

from ..schemas import ToolInput, schema_from_model
from ..config import Settings
from .search import _raw_arxiv_search, _validate_categories

logger = logging.getLogger("arxiv-mcp-pro")
settings = Settings()

WATCH_FILE_NAME = "watched_topics.json"


class CheckAlertsInput(ToolInput):
    """Arguments for the `check_alerts` tool."""

    topic: Optional[str] = Field(
        default=None,
        description=(
            "Optional: check only this specific watched topic (must match the topic string used in watch_topic exactly). Omit to check all saved watches."
        ),
    )


class WatchTopicInput(ToolInput):
    """Arguments for the `watch_topic` tool."""

    categories: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional arXiv category filter (e.g. ['cs.LG', 'cs.AI']). Narrows results to specific fields."
        ),
    )
    max_results: int = Field(
        default=10,
        description=("Maximum papers to return per alert check (default: 10)."),
    )
    topic: str = Field(
        description=(
            'Query string to monitor. Uses arXiv search syntax — quoted phrases for exact matches, field specifiers (ti:, au:, abs:), and boolean operators (AND, OR, ANDNOT). Example: \'"reinforcement learning" AND "robotics"\'.'
        )
    )


watch_topic_tool = types.Tool(
    name="watch_topic",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, openWorldHint=False
    ),
    description=(
        "Save or update a persistent research topic watch. "
        "When checked via check_alerts, returns only papers published since the last check — "
        "acting as a standing alert for new work on a topic. "
        "The topic string uses the same query syntax as search_papers (quoted phrases, field specifiers, boolean operators). "
        'Examples: \'"diffusion models" AND ti:"video generation"\', \'au:"LeCun" AND cs.LG\'. '
        "Calling watch_topic with the same topic string updates the existing watch rather than creating a duplicate. "
        "Pair with check_alerts to poll for new papers."
    ),
    inputSchema=schema_from_model(WatchTopicInput),
)

check_alerts_tool = types.Tool(
    name="check_alerts",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Check all saved topic watches for newly published papers since the last check. "
        "Omitting the topic parameter runs ALL saved watches and returns new papers for each. "
        "Passing a topic string checks only that specific watch. "
        "Updates each watch's last_checked timestamp after a SUCCESSFUL check, so subsequent calls only return newer papers. "
        "Topics are checked independently: if one topic's search fails (e.g. a transient arXiv error or rate limit), "
        "its entry carries an `error` field with `new_paper_count: 0` and its last_checked is left unchanged so it retries "
        "next call, while the other topics still return normally. "
        "Use watch_topic to register topics before calling this. "
        "Returns a summary with per-topic new paper counts and full paper metadata (plus an `error` field for any topic that failed)."
    ),
    inputSchema=schema_from_model(CheckAlertsInput),
)


def _watch_file_path() -> Path:
    """Get watched topics file path."""
    return Path(settings.STORAGE_PATH) / WATCH_FILE_NAME


def _load_watches() -> Dict[str, Any]:
    """Load watch storage from disk."""
    watch_file = _watch_file_path()
    if not watch_file.exists():
        return {"topics": []}

    try:
        return json.loads(watch_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Invalid watched topics file, resetting: %s", watch_file)
        return {"topics": []}


def _save_watches(payload: Dict[str, Any]) -> None:
    """Persist watches to disk atomically (temp file + rename).

    Writing in place risks a truncated or empty file if the process is
    interrupted mid-write; ``_load_watches`` would then discard it and silently
    lose every saved watch. Write to a temp file in the same directory, fsync it,
    then ``os.replace`` — an atomic rename on POSIX and Windows — so a reader
    always sees either the complete old file or the complete new one.
    """
    watch_file = _watch_file_path()
    data = json.dumps(payload, indent=2)
    fd, tmp_path = tempfile.mkstemp(
        dir=watch_file.parent, prefix=f".{WATCH_FILE_NAME}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, watch_file)
    except BaseException:
        # Never leave a stray temp file behind if the write or rename fails.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _now_iso() -> str:
    """UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _filter_by_topic(
    topics: List[Dict[str, Any]], topic_name: Optional[str]
) -> List[Dict[str, Any]]:
    """Filter watched topics by exact topic name if provided."""
    if not topic_name:
        return topics
    return [topic for topic in topics if topic.get("topic") == topic_name]


def _is_new_paper(published_value: str, last_checked: Optional[str]) -> bool:
    """Check if paper is newer than the last check timestamp."""
    if not last_checked:
        return True

    try:
        return parser.parse(published_value) > parser.parse(last_checked)
    except (ValueError, TypeError):
        return True


async def handle_watch_topic(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Save or update a watched topic definition."""
    try:
        topic = (arguments.get("topic") or "").strip()
        if not topic:
            return [types.TextContent(type="text", text="Error: topic is required")]

        categories = arguments.get("categories") or []
        # Reject malformed/injection category values at WRITE time, so a bad value
        # can never be persisted and then replayed unchecked by check_alerts (the
        # _raw_arxiv_search backstop would catch it at poll time, but rejecting at
        # save time gives the caller an immediate, actionable error).
        if categories and not _validate_categories(categories):
            return [
                types.TextContent(
                    type="text",
                    text="Error: Invalid category provided. Please check arXiv category names.",
                )
            ]
        max_results = min(int(arguments.get("max_results", 10)), settings.MAX_RESULTS)

        payload = _load_watches()
        topics = payload.get("topics", [])
        existing_index = next(
            (idx for idx, item in enumerate(topics) if item.get("topic") == topic), None
        )

        record = {
            "topic": topic,
            "categories": categories,
            "max_results": max_results,
            "last_checked": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }

        if existing_index is not None:
            current = topics[existing_index]
            record["created_at"] = current.get("created_at", record["created_at"])
            record["last_checked"] = current.get("last_checked")
            topics[existing_index] = record
        else:
            topics.append(record)

        payload["topics"] = topics
        _save_watches(payload)

        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "success",
                        "message": "Topic watch saved",
                        "topic": record,
                    },
                    indent=2,
                ),
            )
        ]
    except Exception as exc:
        logger.error("watch_topic error: %s", exc)
        return [types.TextContent(type="text", text=f"Error: {str(exc)}")]


async def handle_check_alerts(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Check all watched topics (or one topic) for newly published papers."""
    try:
        selected_topic = (arguments.get("topic") or "").strip() or None
        payload = _load_watches()
        all_topics = payload.get("topics", [])
        topics = _filter_by_topic(all_topics, selected_topic)

        now_iso = _now_iso()
        alerts: List[Dict[str, Any]] = []

        for topic in topics:
            topic_query = topic.get("topic", "")
            if not topic_query:
                continue

            last_checked = topic.get("last_checked")
            max_results = min(int(topic.get("max_results", 10)), settings.MAX_RESULTS)
            try:
                search_results = await _raw_arxiv_search(
                    query=topic_query,
                    max_results=max_results,
                    sort_by="date",
                    date_from=last_checked,
                    categories=topic.get("categories") or None,
                )
            except Exception as exc:
                # One topic's search failure (a transient arXiv error, a rate
                # limit on this N-search loop) must not abort the whole batch or
                # roll back topics that already succeeded. Report the error for
                # this topic, leave its last_checked untouched so it is retried
                # next run, and continue to the next topic. The try wraps only
                # the network call — a malformed record (e.g. a non-int
                # max_results) surfaces loudly via the outer handler rather than
                # being masked here as a transient per-topic error.
                logger.error("check_alerts: topic %r failed: %s", topic_query, exc)
                alerts.append(
                    {
                        "topic": topic_query,
                        "last_checked": last_checked,
                        "error": str(exc),
                        "new_paper_count": 0,
                        "new_papers": [],
                    }
                )
                continue

            new_papers = [
                paper
                for paper in search_results
                if _is_new_paper(paper.get("published", ""), last_checked)
            ]

            alerts.append(
                {
                    "topic": topic_query,
                    "last_checked": last_checked,
                    "new_paper_count": len(new_papers),
                    "new_papers": new_papers,
                }
            )

            topic["last_checked"] = now_iso
            topic["updated_at"] = now_iso
            # Persist each topic's advance as soon as it succeeds. _save_watches
            # is atomic (temp + fsync + os.replace, B5), so saving per checked
            # topic is safe and means a later failure or interrupt cannot make
            # this topic re-report papers it has already seen.
            payload["topics"] = all_topics
            _save_watches(payload)

        result = {
            "status": "success",
            "checked_topics": len(topics),
            "alerts": alerts,
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as exc:
        logger.error("check_alerts error: %s", exc)
        return [types.TextContent(type="text", text=f"Error: {str(exc)}")]
