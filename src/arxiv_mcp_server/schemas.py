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

    An optional *property* is modelled as ``anyOf: [{...T...}, {"type":
    "null"}]``. A client reads a missing key as absent, so the null branch adds
    nothing there.

    This applies to properties only. Inside a collection — `list[int | None]` —
    the null branch is the difference between accepting `[null]` and rejecting
    it, so collapsing it there would advertise a stricter schema than the model
    actually enforces.
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


def _clean_schema(node: Any) -> Any:
    """Drop pydantic bookkeeping that is noise in an MCP tool schema.

    Removes generated `title`s (pydantic derives one per model and per field;
    none tell a client anything the property name does not) and the implicit
    `default: null` an optional field picks up.

    Mappings whose keys are *names* rather than schema keywords — `properties`,
    `$defs` — are recursed into by value, never filtered by key. Otherwise a
    field genuinely named `title` would be deleted from the schema while
    remaining in `required`, leaving a tool that rejects the call whether the
    argument is supplied or not.
    """
    if isinstance(node, list):
        return [_clean_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned: Dict[str, Any] = {}
    for key, value in node.items():
        if key == "title":
            continue
        if key == "default" and value is None:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                name: (
                    _clean_schema(_strip_nullable(sub))
                    if isinstance(sub, dict)
                    else sub
                )
                for name, sub in value.items()
            }
        elif key in ("$defs", "definitions") and isinstance(value, dict):
            cleaned[key] = {name: _clean_schema(sub) for name, sub in value.items()}
        else:
            cleaned[key] = _clean_schema(value)
    return cleaned


def schema_from_model(model: Type[BaseModel]) -> Dict[str, Any]:
    """Return the MCP `inputSchema` for *model*.

    Key order is normalised to the shape these schemas have always had —
    `type`, `properties`, `required`, `additionalProperties` — so a generated
    schema diffs cleanly against the hand-written one it replaces.
    """
    raw = _clean_schema(model.model_json_schema())

    # A self-referential model is emitted as `{"$defs": {...}, "$ref": ...}`
    # with no top-level `properties`. Flattening that would silently advertise
    # an empty closed object, so every real call would be rejected as carrying
    # additional properties. Refuse loudly instead: no tool needs this today,
    # and a clear error beats a schema that is quietly wrong.
    if "$ref" in raw:
        raise ValueError(
            f"{model.__name__} is self-referential; its schema has a root $ref "
            "that schema_from_model cannot flatten into an MCP input schema."
        )

    schema: Dict[str, Any] = {"type": "object", "properties": raw.get("properties", {})}
    if "required" in raw:
        schema["required"] = raw["required"]
    else:
        schema["required"] = []
    schema["additionalProperties"] = raw.get("additionalProperties", False)

    # A nested model, or anything else pydantic factors out, is emitted as a
    # `$ref` into a top-level `$defs`. Rebuilding the schema from a fixed set
    # of keys would drop that block and leave the reference dangling, which a
    # client's validator cannot resolve — the tool would fail before its
    # handler ever ran. No tool nests a model today; this keeps the first one
    # that does from being broken on arrival.
    for key in ("$defs", "definitions"):
        if key in raw:
            schema[key] = raw[key]
    return schema
