"""Tool input models, and the MCP-shaped JSON Schemas derived from them.

Every tool used to carry a hand-written `inputSchema` dict next to a handler
that read `arguments` by key. Nothing tied the two together, so they could
drift silently — and cross-cutting schema work meant editing every tool file by
hand (see `d22255b`, which added `additionalProperties: false` to nine files one
line at a time).

Here the model is the single definition. The schema advertised over MCP is
generated from it, and the handler validates its arguments through the same
model, so a parameter cannot exist in one and not the other.

`ToolInput` closes the schema by default: `extra="forbid"` is what emits
`additionalProperties: false`, so that property can no longer be forgotten on a
new tool.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel, ConfigDict

__all__ = ["ToolInput", "schema_from_model"]


class ToolInput(BaseModel):
    """Base class for a tool's arguments.

    Closed by default: unknown arguments are rejected rather than silently
    ignored, and the generated schema says so.
    """

    model_config = ConfigDict(extra="forbid")


def _strip_nullable(node: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse pydantic's `Optional[T]` encoding back to a plain type.

    An optional field is modelled as ``anyOf: [{...T...}, {"type": "null"}]``.
    MCP clients read a missing key as absent, so the null branch carries no
    information for them and only makes the schema harder to read.
    """
    branches = node.get("anyOf")
    if not isinstance(branches, list):
        return node
    non_null = [b for b in branches if b.get("type") != "null"]
    if len(non_null) != 1:
        return node
    merged = {k: v for k, v in node.items() if k != "anyOf"}
    merged.update(non_null[0])
    return merged


def _clean(node: Any) -> Any:
    """Drop pydantic bookkeeping that is noise in an MCP tool schema.

    Removes generated `title`s (pydantic derives one per model and per field;
    none of them tell a client anything the property name does not) and the
    implicit `default: null` that an optional field picks up.
    """
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node

    node = _strip_nullable(node)
    cleaned: Dict[str, Any] = {}
    for key, value in node.items():
        if key == "title":
            continue
        if key == "default" and value is None:
            continue
        cleaned[key] = _clean(value)
    return cleaned


def schema_from_model(model: Type[BaseModel]) -> Dict[str, Any]:
    """Return the MCP `inputSchema` for *model*.

    Key order is normalised to the shape these schemas have always had —
    `type`, `properties`, `required`, `additionalProperties` — so a generated
    schema diffs cleanly against the hand-written one it replaces.
    """
    raw = _clean(model.model_json_schema())

    schema: Dict[str, Any] = {"type": "object", "properties": raw.get("properties", {})}
    if "required" in raw:
        schema["required"] = raw["required"]
    else:
        schema["required"] = []
    schema["additionalProperties"] = raw.get("additionalProperties", False)
    return schema
