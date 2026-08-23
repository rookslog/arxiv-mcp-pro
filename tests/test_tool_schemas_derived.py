"""The derived schemas must stay identical to the hand-written ones they replaced.

`tests/fixtures/tool_schema_snapshot.json` is a capture of every tool's
advertised schema taken immediately before the models were introduced. It is
the evidence that generating schemas changed nothing a client can observe.

One deliberate normalisation: four tools previously omitted `required`
entirely where the other seven wrote `required: []`. The generated schemas
always emit it. That is semantically identical — an absent `required` and an
empty one both mean "no required properties" — and it removes an inconsistency
between tools.
"""

import json
from pathlib import Path

import pytest

from arxiv_mcp_server.schemas import ToolInput, schema_from_model
from arxiv_mcp_server.tools import (
    abstract_tool,
    citation_graph_tool,
    download_tool,
    library_influence_tool,
    list_tool,
    read_tool,
    reindex_tool,
    search_tool,
    semantic_search_tool,
    watch_topic_tool,
)
from arxiv_mcp_server.tools.alerts import check_alerts_tool

SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures" / "tool_schema_snapshot.json").read_text()
)

ALL_TOOLS = [
    search_tool,
    download_tool,
    list_tool,
    read_tool,
    abstract_tool,
    semantic_search_tool,
    reindex_tool,
    citation_graph_tool,
    library_influence_tool,
    watch_topic_tool,
    check_alerts_tool,
]


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_derived_schema_matches_the_hand_written_one(tool):
    """No client-observable change to any advertised input schema."""
    expected = dict(SNAPSHOT[tool.name]["inputSchema"])
    expected.setdefault("required", [])
    assert tool.inputSchema == expected


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_descriptions_are_unchanged(tool):
    """Tool descriptions are hand-written prose and must survive untouched."""
    assert tool.description == SNAPSHOT[tool.name]["description"]


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_every_schema_is_closed(tool):
    """`additionalProperties: false` is now structural, not per-tool discipline.

    It comes from `ToolInput`'s `extra="forbid"`, so a new tool cannot forget it
    the way nine tools once did (see `d22255b`, which added the property to
    every tool file one line at a time).
    """
    assert tool.inputSchema["additionalProperties"] is False


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_every_property_is_documented(tool):
    """A parameter with no description is invisible to the model calling it."""
    for name, spec in tool.inputSchema["properties"].items():
        assert spec.get("description"), f"{tool.name}.{name} has no description"


def test_snapshot_covers_every_registered_tool():
    """A new tool must be added to the snapshot, not silently skipped."""
    assert {t.name for t in ALL_TOOLS} == set(SNAPSHOT)


def test_closed_schema_is_inherited_not_repeated():
    """A model that declares nothing still produces a closed schema."""

    class Empty(ToolInput):
        pass

    assert schema_from_model(Empty) == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_optional_fields_do_not_leak_pydantic_encoding():
    """`Optional[T]` must advertise as T, not as an anyOf with a null branch."""
    from typing import Optional

    from pydantic import Field

    class Model(ToolInput):
        maybe: Optional[int] = Field(default=None, description="d")

    assert schema_from_model(Model)["properties"]["maybe"] == {
        "type": "integer",
        "description": "d",
    }
