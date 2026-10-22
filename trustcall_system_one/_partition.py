"""Splitting a Clairic request by field type and merging the answers back.

``FieldTypeRoutingTransport`` sends the fields of one Clairic request to different
providers according to their type. Every partition carries the same ``state`` (and
images, reference date and options), so each provider's spans index the same text and
stay exact; the merge keeps the fields in the order they were asked, sums usage and
reconciles ``truncated`` and ``state_text``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from trustcall_system_one._errors import ModelResponseError

__all__ = [
    "FIELD_TYPES",
    "LIST_ITEMS",
    "field_key",
    "merge_bodies",
    "partition_fields",
    "route_keys",
    "sum_usage",
]

FIELD_TYPES: tuple[str, ...] = (
    "text",
    "number",
    "date",
    "time",
    "datetime",
    "duration",
    "list",
    "estimate",
    "forecast",
)
"""The ``type`` of a Clairic field spec (``_wire.ClairicField``)."""
LIST_ITEMS: tuple[str, ...] = ("text", "number", "date")
"""The ``items`` of a list field."""

_SUMMED = (
    "state_tokens",
    "image_tokens",
    "schema_tokens",
    "input_tokens",
    "output_tokens",
    "passes",
)


def route_keys() -> tuple[str, ...]:
    """Every key a field-type route may use: a type, or ``list:<items>``."""
    return (*FIELD_TYPES, *(f"list:{items}" for items in LIST_ITEMS))


def field_key(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """The route keys of a field spec, most specific first.

    ``("list:date", "list")`` for a list of dates, ``("date",)`` for a date.
    """
    kind = str(spec.get("type", ""))
    if kind == "list" and spec.get("items"):
        return (f"list:{spec['items']}", "list")
    return (kind,)


def partition_fields(
    fields: Mapping[str, Mapping[str, Any]],
    pick: Mapping[str, Any],
    default: Any,
) -> list[tuple[Any, list[str]]]:
    """Group field names by the transport their type is routed to.

    Args:
        fields: The request's ``fields`` (name to spec), in the order asked.
        pick: Route key (``text``, ``list:date``, ...) to transport.
        default: The transport of every field whose type has no route.

    Returns:
        ``(transport, field names)`` pairs, in order of the first field each one
        serves; two types routed to the same transport object share a partition,
        and a partition keeps the fields in the order asked.
    """
    groups: list[tuple[Any, list[str]]] = []
    for name, spec in fields.items():
        target = default
        for key in field_key(spec):
            if key in pick:
                target = pick[key]
                break
        for transport, names in groups:
            if transport is target:
                names.append(name)
                break
        else:
            groups.append((target, [name]))
    return groups


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def sum_usage(usages: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Add up the ``usage`` objects of several responses.

    Token counts are summed (a key is kept when any response has it), and so is a
    reported ``cost``.
    """
    total: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    cost: float | None = None
    for usage in usages:
        if not isinstance(usage, Mapping):
            continue
        for key in _SUMMED:
            number = _int(usage.get(key))
            if number is not None:
                total[key] = total.get(key, 0) + number
        spent = usage.get("cost")
        if (
            isinstance(spent, (int, float))
            and not isinstance(spent, bool)
            and spent >= 0
        ):
            cost = (cost or 0.0) + float(spent)
    if cost is not None:
        total["cost"] = cost
    return total


def _bad(why: str) -> ModelResponseError:
    return ModelResponseError(f"FieldTypeRoutingTransport: {why}")


def merge_bodies(
    order: Sequence[str],
    parts: Sequence[tuple[Sequence[str], Mapping[str, Any]]],
    labels: Sequence[Mapping[str, Any]],
    sent: Any = None,
) -> tuple[dict[str, Any], list[str]]:
    """Merge the Clairic response bodies of the partitions of one request.

    Args:
        order: Every field name, in the order the request asked.
        parts: ``(field names, response body)`` per partition.
        labels: One ``{"provider", "served_model", ...}`` per partition, copied into
            the ``routing`` note of the merged body.
        sent: The ``state`` the request carried. A partition that returns no
            ``state_text`` while another does read the state as it was sent.

    Returns:
        ``(body, notes)``: the merged body and what was reconciled on the way (the
        notes become warnings of the response).

    Raises:
        ModelResponseError: If a body is not a Clairic response (the message names
            no field and quotes no text).
    """
    notes: list[str] = []
    answers: dict[str, Any] = {}
    truncated = False
    returned: list[str | None] = []
    for names, body in parts:
        got = body.get("answers")
        if not isinstance(got, Mapping):
            raise _bad("a partition's response has no answers")
        for name in names:
            if name in got:
                answers[name] = got[name]
        truncated = truncated or bool(body.get("truncated"))
        text = body.get("state_text")
        returned.append(text if isinstance(text, str) else None)
    merged: dict[str, Any] = {
        "model": next((str(b["model"]) for _, b in parts if b.get("model")), "clairic"),
        "answers": {name: answers[name] for name in order if name in answers},
        "truncated": truncated,
        "usage": sum_usage([b.get("usage") for _, b in parts]),
        "routing": {"partitions": [dict(label) for label in labels]},
    }
    if any(t is not None for t in returned):
        # a partition that sent no text read the state as it was sent
        fill = sent if isinstance(sent, str) else None
        texts = [t if t is not None else fill for t in returned]
        known = [t for t in texts if t is not None]
        shortest = min(known, key=len)
        if all(t.startswith(shortest) for t in known):
            # equal, or some partition read only the beginning of the input: the
            # shared prefix is the text every span can be mapped through
            merged["state_text"] = shortest
        else:
            first = known[0]
            merged["state_text"] = first
            for (names, _), text in zip(parts, texts, strict=True):
                if text != first:
                    for name in names:
                        entry = merged["answers"].get(name)
                        if isinstance(entry, Mapping):
                            merged["answers"][name] = {
                                k: v
                                for k, v in entry.items()
                                if k not in ("span", "spans")
                            }
            notes.append(
                "field_type_routing: the partitions returned different state "
                "texts; the spans of the fields that disagree were dropped"
            )
    return merged, notes
