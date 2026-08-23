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

import asyncio
import json
from pathlib import Path

import pytest

from arxiv_mcp_server.schemas import ToolInput, schema_from_model
from arxiv_mcp_server.server import list_tools as registered_tools

SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures" / "tool_schema_snapshot.json").read_text()
)

# The server's own `list_tools` handler is the authority on what is registered.
# Deriving from it means a twelfth tool cannot be added to the server and quietly
# skipped by every check in this file — which a second hand-maintained list here
# would have allowed, while still passing its own coverage assertion.
ALL_TOOLS = asyncio.run(registered_tools())


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
    """A new tool must be added to the snapshot, not silently skipped.

    `ALL_TOOLS` comes from the server's own `list_tools`, so this compares the
    snapshot against what the server actually advertises rather than against a
    second list maintained alongside it.
    """
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


def test_nested_models_keep_their_definitions():
    """A `$ref` must not outlive the `$defs` block it points at.

    Pydantic factors a nested model into top-level `$defs` and leaves a
    `$ref` behind. Dropping that block would leave a dangling reference a
    client's validator cannot resolve, failing the call before the handler
    runs. No tool nests a model today — this keeps the first one that does
    from arriving broken.
    """
    from typing import Optional

    from pydantic import BaseModel, Field

    class Child(BaseModel):
        a: int = Field(description="a")

    class Parent(ToolInput):
        child: Optional[Child] = Field(default=None, description="nested")

    schema = schema_from_model(Parent)
    ref = schema["properties"]["child"]["$ref"]

    assert ref == "#/$defs/Child"
    assert "$defs" in schema, "referenced definition was dropped"
    assert schema["$defs"]["Child"]["properties"]["a"]["type"] == "integer"


def test_flat_models_do_not_gain_an_empty_defs_block():
    """The eleven real tools must keep the exact schema they already had."""
    from pydantic import Field

    class Flat(ToolInput):
        a: int = Field(description="a")

    assert "$defs" not in schema_from_model(Flat)


def test_a_field_named_title_survives_cleaning():
    """Stripping pydantic's generated `title` must not delete a field called title.

    `title` is both a schema annotation and a plausible parameter name. Filtering
    it by key everywhere would drop the property while leaving it in `required`,
    producing a tool that rejects the call whether the argument is present (as an
    additional property) or absent (as a missing required one).
    """
    from pydantic import Field

    class Titled(ToolInput):
        title: str = Field(description="the title to use")

    schema = schema_from_model(Titled)

    assert "title" in schema["properties"], "the field was stripped as metadata"
    assert schema["properties"]["title"] == {
        "type": "string",
        "description": "the title to use",
    }
    assert schema["required"] == ["title"]


def test_nullable_collection_elements_keep_their_null_branch():
    """`list[int | None]` accepts `[null]`; the schema must not say otherwise.

    Collapsing nullability is right for an optional property, where omission and
    null mean the same thing to a client. Inside a collection it is not: it would
    advertise a stricter schema than the model enforces, so valid input would be
    rejected before the handler ran.
    """
    from typing import List, Optional

    from pydantic import Field

    class WithList(ToolInput):
        values: List[Optional[int]] = Field(description="values")

    items = schema_from_model(WithList)["properties"]["values"]["items"]

    assert "anyOf" in items, "the null branch was collapsed away"
    assert {b.get("type") for b in items["anyOf"]} == {"integer", "null"}


def test_a_self_referential_model_is_refused_loudly():
    """A recursive model yields a root `$ref` that cannot be flattened.

    Flattening it silently would advertise an empty closed object and reject
    every real call. Failing at definition time beats failing at call time.
    """
    from typing import Optional

    import pytest
    from pydantic import Field

    class Node(ToolInput):
        value: int = Field(description="v")
        child: Optional["Node"] = Field(default=None, description="c")

    Node.model_rebuild()

    with pytest.raises(ValueError, match="self-referential"):
        schema_from_model(Node)
