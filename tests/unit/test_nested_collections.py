"""Nested collections (``list[Model]``, ``dict[str, X]``) through the collection engine.

Inserts: the items of one parent, scoped to it (discovery, qualification and the
focus header name the parent), assembled into the parent's document. Updates: the
parent's items are the existing records the discovered items are matched to, and
the result is operations inside the parent's patch list.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from collections import Counter
from typing import Annotated, Any, Optional

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import (
    ExtractionResult,
    Many,
    One,
    RepairConfig,
    ResultRecord,
    create_extractor,
)
from trustcall_system_one.testing import FakeTransport, FakeWorld, route

Doc = dict[str, Any]


class Pet(BaseModel):
    """A pet."""

    name: str = Field(description="the pet's name")
    species: Optional[str] = Field(None, description="what kind of animal it is")


class Person(BaseModel):
    """A person."""

    name: str = Field(description="full name")
    pets: Annotated[list[Pet], Many(identity="name")] = Field(
        default_factory=list, description="their pets"
    )


class Student(BaseModel):
    """A student."""

    name: str = Field(description="full name")
    scores: dict[str, int] = Field(
        default_factory=dict, description="exam score per subject"
    )


class Mug(BaseModel):
    """A product in stock."""

    quantity: int = Field(ge=0, description="units in stock")
    colour: Optional[str] = Field(None, description="the colour")


class Shop(BaseModel):
    """A shop."""

    name: str = Field(description="the shop's name")
    stock: dict[str, Mug] = Field(default_factory=dict, description="stock by product")


def kinds(result: ExtractionResult) -> Counter[str]:
    return Counter(call.kind for call in result.calls)


def ops(record: ResultRecord) -> list[tuple[str, str, Any]]:
    return [(p.op, p.path, p.value) for p in record.patches]


def calls_of(world: FakeWorld, kind: str) -> list[Any]:
    return [c for c in world.transport().calls if c[2].kind == kind]


# --------------------------------------------------------------------------- #
# Insert: the items of one parent
# --------------------------------------------------------------------------- #

OWNERS = (
    "Emma Thompson walks her dog Rex every morning. "
    "Michael Chen keeps a cat called Tom and a parrot named Kiwi."
)


def owners_world() -> FakeWorld:
    world = FakeWorld(OWNERS)
    for name in ("Emma Thompson", "Michael Chen"):
        world.entity(name, schema=Person, mentions=[name], fields={"name": name})
    world.entity(
        "Rex",
        schema=Pet,
        parent="Emma Thompson",
        mentions=["Rex"],
        fields={"name": "Rex", "species": "dog"},
    )
    world.entity(
        "Tom",
        schema=Pet,
        parent="Michael Chen",
        mentions=["Tom"],
        fields={"name": "Tom", "species": "cat"},
    )
    world.entity(
        "Kiwi",
        schema=Pet,
        parent="Michael Chen",
        mentions=["Kiwi"],
        fields={"name": "Kiwi", "species": "parrot"},
    )
    return world


async def test_the_pets_of_each_owner_go_into_that_owners_record() -> None:
    world = owners_world()
    extractor = create_extractor([Many(Person)], transport=world.transport())
    result = await extractor.ainvoke(OWNERS)
    emma, michael = result.records
    assert emma.after == {
        "name": "Emma Thompson",
        "pets": [{"name": "Rex", "species": "dog"}],
    }
    assert michael.after == {
        "name": "Michael Chen",
        "pets": [
            {"name": "Tom", "species": "cat"},
            {"name": "Kiwi", "species": "parrot"},
        ],
    }
    assert [type(p).__name__ for p in michael.value.pets] == ["Pet", "Pet"]  # type: ignore[union-attr]
    assert result.skipped == () and result.warnings == ()
    # the items' answers are answers of the parent, by path
    assert list(michael.fields) == [
        "name",
        "pets.0.name",
        "pets.0.species",
        "pets.1.name",
        "pets.1.species",
    ]
    assert michael.fields["pets.1.name"].path == ("pets", 1, "name")
    assert michael.fields["pets.1.name"].value == "Kiwi"
    # no record of its own for an item
    assert [r.schema_name for r in result.records] == ["Person", "Person"]
    assert len(result.responses) == 2


async def test_discovery_for_a_parent_asks_about_the_items_of_that_parent() -> None:
    world = owners_world()
    extractor = create_extractor([Many(Person)], transport=world.transport())
    await extractor.ainvoke(OWNERS)
    pet_calls = [c for c in world.transport().calls if c[2].schema == "Pet"]
    assert {c[2].scope for c in pet_calls} == {"e0/pets", "e1/pets"}
    discover = [c for c in pet_calls if c[2].kind == "discover"]
    by_scope = {c[2].scope: c[1] for c in discover}
    emma = by_scope["e0/pets"]
    assert emma["state"].startswith(
        '<focus schema="Pet" task="discover" parent="Emma Thompson">'
    )
    assert "of the parent named above" in emma["state"]
    assert "The parent is mentioned in the input at characters 0-13." in emma["state"]
    assert emma["fields"]["mentions"]["instructions"].startswith(
        "Every mention of a Pet belonging to Emma Thompson by their name"
    )
    # presence, count and qualification name the parent too
    presence = [c for c in pet_calls if c[2].kind == "presence"]
    assert all(
        "belonging to `parent.label`" in str(c[1]["questions"]) for c in presence
    )
    assert {c[1]["state"]["parent"]["label"] for c in presence} == {
        "Emma Thompson",
        "Michael Chen",
    }
    (count,) = [
        c for c in pet_calls if c[2].kind == "count" and c[2].scope == "e1/pets"
    ]
    assert "belonging to Michael Chen" in count[1]["fields"]["count"]["instructions"]
    for _, payload, meta in pet_calls:
        if meta.kind == "qualify":
            question = next(iter(payload["questions"].values()))
            assert question["instructions"].endswith("belonging to `parent.label`?")
            assert "parent" in payload["state"]


async def test_the_item_pipeline_is_scoped_to_its_parent() -> None:
    world = owners_world()
    extractor = create_extractor([Many(Person)], transport=world.transport())
    await extractor.ainvoke(OWNERS)
    extracts = [
        c
        for c in world.transport().calls
        if c[2].kind == "extract" and c[2].schema == "Pet"
    ]
    assert len(extracts) == 3  # Rex, Tom, Kiwi: one request each
    header = extracts[0][1]["state"].split("</focus>")[0]
    assert 'entity="Rex" parent="Emma Thompson"' in header
    assert "ignore any pet that belongs to someone else" in header
    attribution = [c for c in world.transport().calls if c[2].kind == "attribution"]
    states = [c[1]["state"] for c in attribution if c[2].schema == "Pet"]
    assert all(s["entity"]["parent"]["label"] for s in states)


async def test_an_item_of_another_owner_is_not_qualified_for_this_one() -> None:
    """Emma's dog is offered for Michael: qualification refuses it."""
    world = owners_world()
    world.bleed_item("Michael Chen", "Rex")
    extractor = create_extractor([Many(Person)], transport=world.transport())
    result = await extractor.ainvoke(OWNERS)
    emma, michael = result.records
    assert [p["name"] for p in emma.after["pets"]] == ["Rex"]  # type: ignore[index]
    assert [p["name"] for p in michael.after["pets"]] == ["Tom", "Kiwi"]  # type: ignore[index]
    (skip,) = result.skipped
    assert (skip.schema_name, skip.reason, skip.entity.label) == (
        "Pet",
        "not_qualified",
        "Rex",
    )
    assert skip.entity.parent == "e1"  # Michael's key
    # the count (2) agrees with the pets that qualify, so nothing is broadened
    assert result.warnings == ()


async def test_a_one_schema_scopes_its_items_to_the_one_parent() -> None:
    text = "Emma Thompson walks her dog Rex."
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
    )
    world.entity(
        "Rex",
        schema=Pet,
        parent="the Person",
        mentions=["Rex"],
        fields={"name": "Rex", "species": "dog"},
    )
    extractor = create_extractor([One(Person)], transport=world.transport())
    result = await extractor.ainvoke(text)
    (record,) = result.records
    assert record.after == {
        "name": "Emma Thompson",
        "pets": [{"name": "Rex", "species": "dog"}],
    }
    (discover,) = calls_of(world, "discover")  # the One Person has no discovery
    assert 'parent="the Person"' in discover[1]["state"]
    assert discover[2].scope == "e0/pets"


async def test_no_pets_mentioned_asks_nothing_about_them() -> None:
    text = "Emma Thompson lives in Paris."
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
    )
    extractor = create_extractor([Many(Person)], transport=world.transport())
    result = await extractor.ainvoke(text)
    (record,) = result.records
    assert record.after == {"name": "Emma Thompson", "pets": []}
    pets = [c for c in world.transport().calls if c[2].schema == "Pet"]
    # presence says no: the discovery and count requests were already in flight,
    # but no entity was qualified, matched or extracted
    assert {c[2].kind for c in pets} == {"presence", "discover", "count"}


async def test_a_required_list_nobody_mentions_is_empty_not_null() -> None:
    class Team(BaseModel):
        """A team."""

        name: str = Field(description="the team's name")
        members: list[Pet]

    text = "The Falcons are a team."
    world = FakeWorld(text)
    world.entity(
        "Falcons", schema=Team, mentions=["Falcons"], fields={"name": "Falcons"}
    )
    extractor = create_extractor([Team], transport=world.transport())
    result = await extractor.ainvoke(text)
    (record,) = result.records
    assert record.valid and record.after == {"name": "Falcons", "members": []}


async def test_warnings_of_a_nested_list_name_the_parent() -> None:
    world = owners_world()
    world.count(Pet, 3)  # the model counts a pet more than it finds
    extractor = create_extractor([Many(Person)], transport=world.transport())
    result = await extractor.ainvoke(OWNERS)
    mentions = [w for w in result.warnings if "count_exceeds_discovered" in w]
    assert any(
        w.startswith("Emma Thompson (e0) \u203a pets: count_exceeds_discovered")
        for w in mentions
    )
    assert any(
        w.startswith("Michael Chen (e1) \u203a pets: count_exceeds_discovered")
        for w in mentions
    )


# --------------------------------------------------------------------------- #
# Insert: mappings
# --------------------------------------------------------------------------- #

GRADES = "Dana got 90 in math and 72 in physics."


def grades_world() -> FakeWorld:
    world = FakeWorld(GRADES)
    world.entity("Dana", schema=Student, mentions=["Dana"], fields={"name": "Dana"})
    for key, value in (("math", 90), ("physics", 72)):
        world.entity(
            key,
            schema="StudentScoresEntry",
            parent="Dana",
            mentions=[key],
            fields={"key": key, "value": value},
        )
    return world


async def test_a_mapping_of_primitives_is_keyed_by_the_extracted_key() -> None:
    world = grades_world()
    extractor = create_extractor([Many(Student)], transport=world.transport())
    result = await extractor.ainvoke(GRADES)
    (record,) = result.records
    assert record.after == {"name": "Dana", "scores": {"math": 90, "physics": 72}}
    assert list(record.fields) == ["name", "scores.math", "scores.physics"]
    assert record.fields["scores.math"].path == ("scores", "math")
    assert record.fields["scores.math"].value == 90
    discover = calls_of(world, "discover")[1]
    assert discover[1]["fields"]["mentions"]["instructions"].startswith(
        "Every mention of a score belonging to Dana by their key"
    )


async def test_a_mapping_of_models_nests_the_value_model() -> None:
    text = "Maple Mugs stocks 14 mugs, the blue ones, and 3 plates."
    world = FakeWorld(text)
    world.entity(
        "Maple Mugs",
        schema=Shop,
        mentions=["Maple Mugs"],
        fields={"name": "Maple Mugs"},
    )
    world.entity(
        "mugs",
        schema="Mug",
        parent="Maple Mugs",
        mentions=["mugs"],
        fields={"key": "mugs", "value": {"quantity": 14, "colour": "blue"}},
    )
    world.entity(
        "plates",
        schema="Mug",
        parent="Maple Mugs",
        mentions=["plates"],
        fields={"key": "plates", "value": {"quantity": 3}},
    )
    extractor = create_extractor([Many(Shop)], transport=world.transport())
    result = await extractor.ainvoke(text)
    (record,) = result.records
    assert record.valid, record.errors
    assert record.after == {
        "name": "Maple Mugs",
        "stock": {
            "mugs": {"quantity": 14, "colour": "blue"},
            "plates": {"quantity": 3, "colour": None},
        },
    }
    assert isinstance(record.value.stock["mugs"], Mug)  # type: ignore[union-attr]
    assert record.fields["stock.mugs.quantity"].value == 14
    assert record.fields["stock.plates.quantity"].path == (
        "stock",
        "plates",
        "quantity",
    )


async def test_a_repeated_key_keeps_the_first_entry() -> None:
    world = FakeWorld("Dana got 90 in math and 80 in math again.")
    world.entity("Dana", schema=Student, mentions=["Dana"], fields={"name": "Dana"})
    world.entity(
        "math",
        schema="StudentScoresEntry",
        parent="Dana",
        mentions=["math"],
        fields={"key": "math", "value": 90},
    )
    extractor = create_extractor([Many(Student)], transport=world.transport())
    result = await extractor.ainvoke(world.text)
    (record,) = result.records
    assert record.after == {"name": "Dana", "scores": {"math": 90}}


# --------------------------------------------------------------------------- #
# Depth and recursion
# --------------------------------------------------------------------------- #


async def test_a_list_of_itself_unrolls_to_max_recursion_and_no_further() -> None:
    class Node(BaseModel):
        """A tree node."""

        label: str = Field(description="the node's label")
        children: list["Node"] = Field(default_factory=list, description="sub-nodes")

    Node.model_rebuild()
    text = "Root has Alpha, Alpha has Beta, Beta has Gamma."
    world = FakeWorld(text)
    world.entity("Root", schema=Node, mentions=["Root"], fields={"label": "Root"})
    for label, parent in (("Alpha", "the Node"), ("Beta", "Alpha"), ("Gamma", "Beta")):
        world.entity(
            label,
            schema=Node,
            parent=parent,
            mentions=[(label, 0)],
            fields={"label": label},
        )
    extractor = create_extractor([One(Node)], transport=world.transport())
    result = await extractor.ainvoke(text)
    (record,) = result.records
    # max_recursion=2: the root and two further levels; Gamma is not looked for
    assert record.after == {
        "label": "Root",
        "children": [
            {"label": "Alpha", "children": [{"label": "Beta", "children": []}]}
        ],
    }
    assert sum("max_recursion" in w for w in result.warnings) == 1


async def test_max_depth_keeps_a_deep_collection_from_being_extracted() -> None:
    from trustcall_system_one import Limits

    world = owners_world()
    extractor = create_extractor(
        [Many(Person)], transport=world.transport(), limits=Limits(max_depth=1)
    )
    result = await extractor.ainvoke(OWNERS)
    assert all(r.after["pets"] == [] for r in result.records)  # type: ignore[index]
    assert any("Person: pets: max_depth" in w for w in result.warnings)
    assert not [c for c in world.transport().calls if c[2].schema == "Pet"]


# --------------------------------------------------------------------------- #
# Repair with item paths
# --------------------------------------------------------------------------- #


class Counted(BaseModel):
    """A product line."""

    name: str = Field(description="product name")
    quantity: int = Field(ge=1, description="units")
    note: Optional[str] = Field(None, description="a remark")


class Basket(BaseModel):
    """A basket."""

    owner: str = Field(description="whose basket")
    lines: Annotated[list[Counted], Many(identity="name")] = Field(
        default_factory=list, description="the product lines"
    )


async def test_an_item_that_fails_validation_is_left_out_with_an_item_path() -> None:
    text = "Ines has 0 apples and 3 pears in her basket."
    world = FakeWorld(text)
    world.entity("Ines", schema=Basket, mentions=["Ines"], fields={"owner": "Ines"})
    world.entity(
        "apples",
        schema=Counted,
        parent="Ines",
        mentions=["apples"],
        fields={"name": "apples", "quantity": 0},  # ge=1: never valid
    )
    world.entity(
        "pears",
        schema=Counted,
        parent="Ines",
        mentions=["pears"],
        fields={"name": "pears", "quantity": 3},
    )
    extractor = create_extractor([Many(Basket)], transport=world.transport())
    result = await extractor.ainvoke(text)
    (record,) = result.records
    assert record.valid
    assert record.after == {
        "owner": "Ines",
        "lines": [{"name": "pears", "quantity": 3, "note": None}],
    }
    # the repair loop re-asked the failing field of the item, scoped to it
    repairs = [c for c in result.calls if c.kind == "repair"]
    assert repairs and {c.scope for c in repairs} == {"e0/lines"}
    # the invalid item is reported on the parent, and as a skipped entity
    assert [
        (r.path, r.reason) for r in record.rejected if r.reason == "invalid_item"
    ] == [("/lines", "invalid_item")]
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label, skip.entity.parent) == (
        "validation_failed",
        "apples",
        "e0",
    )
    assert "quantity" in skip.detail
    # the surviving item's answers are keyed by its position in the record
    assert record.fields["lines.0.quantity"].value == 3


async def test_a_failing_optional_field_of_an_item_is_dropped_at_its_item_path() -> (
    None
):
    class Dated(BaseModel):
        """A product line."""

        name: str = Field(description="product name")
        weight: Optional[int] = Field(None, ge=1, description="weight in grams")

    class Crate(BaseModel):
        """A crate."""

        label: str = Field(description="the crate label")
        lines: Annotated[list[Dated], Many(identity="name")] = []

    text = "Crate C1 holds a feather weighing 0 grams."
    world = FakeWorld(text)
    world.entity("C1", schema=Crate, mentions=["C1"], fields={"label": "C1"})
    world.entity(
        "feather",
        schema=Dated,
        parent="C1",
        mentions=["feather"],
        fields={"name": "feather", "weight": 0},
    )
    extractor = create_extractor([Many(Crate)], transport=world.transport())
    result = await extractor.ainvoke(text)
    (record,) = result.records
    assert record.valid
    assert record.after == {
        "label": "C1",
        "lines": [{"name": "feather", "weight": None}],
    }
    assert [
        (r.path, r.reason) for r in record.rejected if r.reason == "validation"
    ] == [("/lines/0/weight", "validation")]
    assert record.fields["lines.0.weight"].value is None
    assert any("repair[" in w for w in record.fields["lines.0.weight"].warnings)


# --------------------------------------------------------------------------- #
# Update: the parent's items are the existing records
# --------------------------------------------------------------------------- #


class LineItem(BaseModel):
    """One product line on an order."""

    name: str = Field(description="product name")
    quantity: int = Field(ge=1, description="number of units ordered")
    sku: Optional[str] = Field(None, description="product code")


class Order(BaseModel):
    """A customer order."""

    order_id: str
    customer: str
    items: Annotated[list[LineItem], Many(identity="name")]


EMAIL = (
    "Hi Sam, it's Ines about order SO-10442. Please add 2 of the blue mugs and "
    "drop the quantity of the tea towels to 1. Everything else stays as it is."
)
ORDER: Doc = {
    "order_id": "SO-10442",
    "customer": "Ines Romero",
    "items": [
        {"name": "Steel kettle", "quantity": 1, "sku": "KTL-STL"},
        {"name": "Tea towels", "quantity": 4, "sku": "TWL-LIN"},
        {"name": "Wooden spoon set", "quantity": 2, "sku": "SPN-WD"},
    ],
}


def order_world(
    text: str = EMAIL, *, mugs: bool = True, towels: bool = True
) -> FakeWorld:
    """The email of the example: tea towels to 1, and blue mugs are new."""
    world = FakeWorld(text)
    world.entity(
        "the Order",
        schema=Order,
        mentions=["SO-10442"],
        fields={"order_id": "SO-10442"},
    )
    if towels:
        world.entity(
            "tea towels",
            schema=LineItem,
            parent="Order",
            mentions=["tea towels"],
            fields={"name": "tea towels", "quantity": 1},
            matches={"name": "Tea towels"},
        )
    if mugs:
        world.entity(
            "blue mugs",
            schema=LineItem,
            parent="Order",
            mentions=["blue mugs"],
            fields={"name": "blue mugs", "quantity": 2},
        )
    return world


def add_line(
    world: FakeWorld, name: str, record_name: str | None, **fields: Any
) -> None:
    """A line of the order that the text names (``record_name``: its existing line)."""
    world.entity(
        name,
        schema=LineItem,
        parent="Order",
        mentions=[name],
        fields={"name": name, **fields},
        matches={"name": record_name} if record_name else None,
    )


async def update_order(
    world: FakeWorld, order: Doc = ORDER, *, inserts: bool = True, **options: Any
) -> ExtractionResult:
    extractor = create_extractor(
        [One(Order)], transport=world.transport(), enable_inserts=inserts, **options
    )
    return await extractor.ainvoke(world.text, existing=[("SO-10442", "Order", order)])


async def test_an_item_is_changed_and_another_added_inside_the_parents_patch() -> None:
    world = order_world()
    result = await update_order(world)
    (record,) = result.records
    assert (record.kind, record.json_doc_id, record.valid) == (
        "update",
        "SO-10442",
        True,
    )
    assert ops(record) == [
        ("replace", "/items/1/quantity", 1),
        ("add", "/items/-", {"name": "blue mugs", "quantity": 2}),
    ]
    assert record.after is not None and record.before == ORDER
    assert record.after["items"][0] == ORDER["items"][0]  # untouched
    assert record.after["items"][2] == ORDER["items"][2]
    assert record.after["items"][1] == {**ORDER["items"][1], "quantity": 1}
    assert record.after["items"][3]["name"] == "blue mugs"
    assert record.after["customer"] == "Ines Romero"
    assert isinstance(record.value.items[3], LineItem)  # type: ignore[union-attr]
    # both operations were gated: the item's change against the item's record, the
    # new item as an addition to the order
    assert record.fields["items.1.quantity"].gate_p == pytest.approx(0.9)
    assert record.fields["items.3.name"].gate_p == pytest.approx(0.9)
    gates = calls_of(world, "gate")
    assert sorted(g[2].scope for g in gates) == ["", "e0/items"]
    item_gate = next(g for g in gates if g[2].scope)
    assert item_gate[1]["state"]["record"] == ORDER["items"][1]  # the item's record
    assert item_gate[1]["state"]["entity"]["parent"]["label"] == "the Order"
    assert item_gate[1]["state"]["proposed_change"]["path"] == "/quantity"
    parent_gate = next(g for g in gates if not g[2].scope)
    assert parent_gate[1]["state"]["proposed_change"]["path"] == "/items/-"


async def test_the_call_sequence_of_an_item_update() -> None:
    world = order_world()
    result = await update_order(world)
    seq = [c.kind for c in result.calls]
    # the order itself first, then the items of that order
    assert seq[:3] == ["presence", "extract", "crosscheck"]
    assert seq.index("discover") > seq.index("crosscheck")
    assert {"discover", "count", "presence", "qualify", "match"} <= set(seq)
    assert kinds(result)["match"] == 2  # one per discovered item
    assert "delete" not in seq


async def test_items_the_text_does_not_mention_are_not_asked_about() -> None:
    world = order_world()
    await update_order(world)
    extracts = [
        c
        for c in world.transport().calls
        if c[2].kind == "extract" and c[2].scope and "<existing" in c[1]["state"]
    ]
    assert len(extracts) == 1  # the tea towels: the one item that has a record
    existing = extracts[0][1]["state"].split("<existing")[1].split("</existing>")[0]
    assert "Tea towels" in existing
    assert "Steel kettle" not in existing and "Wooden spoon" not in existing
    # and nobody asked the matching question about the untouched lines
    for _, payload, meta in world.transport().calls:
        if meta.kind == "match":
            assert "Steel kettle" in str(payload["questions"])  # offered as options
            assert payload["state"]["entity"]["label"] in ("tea towels", "blue mugs")


async def test_a_parent_without_items_takes_them_without_enable_inserts() -> None:
    world = order_world()
    result = await update_order(world, {**ORDER, "items": []}, inserts=False)
    (record,) = result.records
    assert ops(record) == [
        ("add", "/items/-", {"name": "blue mugs", "quantity": 2}),
        ("add", "/items/-", {"name": "tea towels", "quantity": 1}),
    ]
    assert result.skipped == ()
    assert [i["name"] for i in record.after["items"]] == ["blue mugs", "tea towels"]  # type: ignore[index]
    assert kinds(result)["match"] == 0  # nothing to match against


async def test_a_list_the_record_lacks_is_created_for_the_first_item() -> None:
    world = order_world(towels=False)
    without = {"order_id": "SO-10442", "customer": "Ines Romero"}
    result = await update_order(world, without)
    (record,) = result.records
    assert [(p.op, p.path) for p in record.patches] == [
        ("add", "/items"),
        ("add", "/items/-"),
    ]
    assert record.after is not None and len(record.after["items"]) == 1


async def test_a_new_item_is_skipped_without_enable_inserts() -> None:
    world = order_world()
    result = await update_order(world, inserts=False)
    (record,) = result.records
    assert ops(record) == [("replace", "/items/1/quantity", 1)]
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label, skip.entity.parent) == (
        "inserts_disabled",
        "blue mugs",
        "e0",
    )


REMOVAL = (
    "About order SO-10442: Ines no longer wants the steel kettle or the wooden "
    "spoon set, and the tea towels go down to 1."
)


def removal_world() -> FakeWorld:
    world = FakeWorld(REMOVAL)
    world.entity(
        "the Order",
        schema=Order,
        mentions=["SO-10442"],
        fields={"order_id": "SO-10442"},
    )
    # the kettle and the spoons are flagged by the removal question; the text
    # names no entity for them (an entity matched to a flagged line is a conflict)
    add_line(world, "tea towels", "Tea towels", quantity=1)
    world.delete("0", parent="Order").delete("2", parent="Order")
    return world


async def test_items_are_not_removed_unless_enable_item_removals_is_on() -> None:
    world = removal_world()
    result = await update_order(world)
    (record,) = result.records
    assert ops(record) == [("replace", "/items/1/quantity", 1)]
    assert not calls_of(world, "delete")  # not even asked


async def test_removed_items_go_high_index_first() -> None:
    world = removal_world()
    result = await update_order(world, enable_item_removals=True)
    (record,) = result.records
    assert [(p.op, p.path) for p in record.patches] == [
        ("replace", "/items/1/quantity"),
        ("remove", "/items/2"),  # the highest index first: indexes do not shift
        ("remove", "/items/0"),
    ]
    assert record.after is not None
    assert [i["name"] for i in record.after["items"]] == ["Tea towels"]
    assert result.deleted == ()  # an item is not a record: nothing is "deleted"
    assert result.skipped == ()
    deletes = calls_of(world, "delete")
    assert deletes and deletes[0][2].scope == "e0/items"
    question = next(iter(deletes[0][1]["questions"].values()))
    assert "from `parent.label`" in question["instructions"]
    assert deletes[0][1]["state"]["parent"]["label"] == "the Order"


async def test_an_item_that_is_matched_and_flagged_for_removal_is_a_conflict() -> None:
    text = "Order SO-10442: the tea towels should be dropped, no wait, make them 1."
    world = FakeWorld(text)
    world.entity(
        "the Order",
        schema=Order,
        mentions=["SO-10442"],
        fields={"order_id": "SO-10442"},
    )
    add_line(world, "tea towels", "Tea towels", quantity=1)
    world.delete("1", parent="Order")
    result = await update_order(world, enable_item_removals=True)
    (record,) = result.records
    assert record.patches == ()  # neither updated nor removed
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label, skip.entity.parent) == (
        "conflict_update_delete",
        "tea towels",
        "e0",
    )
    assert skip.candidates and skip.candidates[0][0] == "1"


async def test_the_identity_guard_refuses_one_items_update_only() -> None:
    text = "Order SO-10442: make the bath mats 1, and the steel kettle 3."
    world = FakeWorld(text)
    world.entity(
        "the Order",
        schema=Order,
        mentions=["SO-10442"],
        fields={"order_id": "SO-10442"},
    )
    # Clef claims the "bath mats" are the "Tea towels" line: a confident wrong match
    add_line(world, "bath mats", "Tea towels", quantity=1)
    add_line(world, "steel kettle", "Steel kettle", quantity=3)
    result = await update_order(world)
    (record,) = result.records
    # the kettle's update stands; the bath mats' is refused as a whole
    assert ops(record) == [("replace", "/items/0/quantity", 3)]
    refused = [r for r in record.rejected if r.reason == "identity_conflict"]
    # every operation of that item, the rename included
    assert [(r.op, r.path) for r in refused] == [
        ("replace", "/items/1/name"),
        ("replace", "/items/1/quantity"),
    ]
    assert "bath mats" in refused[0].detail and "Tea towels" in refused[0].detail
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label, skip.entity.parent) == (
        "ambiguous_match",
        "bath mats",
        "e0",
    )
    assert skip.detail.startswith("identity conflict:")
    assert record.after is not None
    assert record.after["items"][1] == ORDER["items"][1]  # the tea towels: untouched


async def test_two_more_of_an_existing_line_is_read_as_the_stated_value() -> None:
    """Relative edits are not supported: nothing is computed from the old value."""
    text = "Order SO-10442: Ines wants two more of the tea towels."
    world = FakeWorld(text)
    world.entity(
        "the Order",
        schema=Order,
        mentions=["SO-10442"],
        fields={"order_id": "SO-10442"},
    )
    add_line(world, "tea towels", "Tea towels", quantity=2)  # what the text states
    result = await update_order(world)
    (record,) = result.records
    # 4 + 2 = 6 is NOT computed: the extraction reads "two", and the gate decides
    assert ops(record) == [("replace", "/items/1/quantity", 2)]


async def test_an_error_the_record_already_had_is_not_the_updates_to_repair() -> None:
    broken: Doc = {
        **ORDER,
        "items": [
            {"name": "Steel kettle", "quantity": 0, "sku": "KTL-STL"},  # ge=1
            *ORDER["items"][1:],
        ],
    }
    world = order_world()
    result = await update_order(world, broken)
    (record,) = result.records
    # the update's own operations stay; the record's old error is reported, not
    # blamed on them, and costs no repair request
    assert ("replace", "/items/1/quantity", 1) in ops(record)
    assert not record.valid
    assert any("items.0.quantity" in e for e in record.errors)
    assert kinds(result)["repair"] == 0


async def test_an_operation_that_breaks_the_parent_is_dropped_with_its_path() -> None:
    class Strict(BaseModel):
        """An order that holds at most three lines."""

        order_id: str
        items: Annotated[list[LineItem], Many(identity="name")] = Field(max_length=3)

    world = FakeWorld(EMAIL)
    world.entity(
        "the Strict",
        schema=Strict,
        mentions=["SO-10442"],
        fields={"order_id": "SO-10442"},
    )
    world.entity(
        "blue mugs",
        schema=LineItem,
        parent="Strict",
        mentions=["blue mugs"],
        fields={"name": "blue mugs", "quantity": 2},
    )
    record_in = {k: v for k, v in ORDER.items() if k != "customer"}
    for repair in (RepairConfig(enabled=False), RepairConfig()):
        extractor = create_extractor(
            [One(Strict)],
            transport=world.transport(),
            enable_inserts=True,
            repair=repair,
        )
        result = await extractor.ainvoke(
            world.text, existing=[("SO-10442", "Strict", record_in)]
        )
        (record,) = result.records
        assert record.patches == ()  # the add would make four lines
        dropped = [r for r in record.rejected if r.reason == "validation" and r.op]
        assert [(r.op, r.path) for r in dropped] == [("add", "/items/-")]
        assert record.valid and record.after == record.before
        # a collection is not a field the repair loop can re-ask: no request
        assert kinds(result)["repair"] == 0


# --------------------------------------------------------------------------- #
# Update: Many parents, mappings
# --------------------------------------------------------------------------- #

KENNEL = (
    "Emma Thompson now has a second dog, Luna. "
    "Michael Chen's cat Tom turned out to be a kitten."
)


async def test_pets_are_added_to_and_changed_in_the_right_owners_record() -> None:
    world = FakeWorld(KENNEL)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={"name": "Michael Chen"},
        matches={"name": "Michael Chen"},
    )
    world.entity(
        "Luna",
        schema=Pet,
        parent="Emma Thompson",
        mentions=["Luna"],
        fields={"name": "Luna", "species": "dog"},
    )
    world.entity(
        "Tom",
        schema=Pet,
        parent="Michael Chen",
        mentions=["Tom"],
        fields={"name": "Tom", "species": "kitten"},
        matches={"name": "Tom"},
    )
    existing = [
        (
            "1",
            "Person",
            {"name": "Emma Thompson", "pets": [{"name": "Rex", "species": "dog"}]},
        ),
        (
            "2",
            "Person",
            {"name": "Michael Chen", "pets": [{"name": "Tom", "species": "cat"}]},
        ),
    ]
    extractor = create_extractor(
        [Many(Person)], transport=world.transport(), enable_inserts=True
    )
    result = await extractor.ainvoke(KENNEL, existing=existing)
    emma, michael = result.records
    assert ops(emma) == [("add", "/pets/-", {"name": "Luna", "species": "dog"})]
    assert ops(michael) == [("replace", "/pets/0/species", "kitten")]
    assert [p["name"] for p in emma.after["pets"]] == ["Rex", "Luna"]  # type: ignore[index]
    assert emma.fields["pets.1.name"].path == ("pets", 1, "name")
    assert result.skipped == () and result.warnings == ()


GRADE_UPDATE = "Dana's math score is now 95, and she got 80 in chemistry."


def student_world() -> FakeWorld:
    world = FakeWorld(GRADE_UPDATE)
    world.entity(
        "Dana",
        schema=Student,
        mentions=["Dana"],
        fields={"name": "Dana"},
        matches={"name": "Dana"},
    )
    world.entity(
        "math",
        schema="StudentScoresEntry",
        parent="Dana",
        mentions=["math"],
        fields={"key": "math", "value": 95},
        matches={"key": "math"},
    )
    world.entity(
        "chemistry",
        schema="StudentScoresEntry",
        parent="Dana",
        mentions=["chemistry"],
        fields={"key": "chemistry", "value": 80},
    )
    return world


DANA = ("d", "Student", {"name": "Dana", "scores": {"math": 90, "physics": 72}})


async def test_a_mapping_entry_is_replaced_added_and_the_rest_left_alone() -> None:
    world = student_world()
    extractor = create_extractor(
        [Many(Student)], transport=world.transport(), enable_inserts=True
    )
    result = await extractor.ainvoke(GRADE_UPDATE, existing=[DANA])
    (record,) = result.records
    assert ops(record) == [
        ("replace", "/scores/math", 95),
        ("add", "/scores/chemistry", 80),
    ]
    assert record.after == {
        "name": "Dana",
        "scores": {"math": 95, "physics": 72, "chemistry": 80},
    }
    assert record.fields["scores.math"].gate_p == pytest.approx(0.9)
    # a mapping key is never renamed
    assert not [r for r in record.rejected if r.reason == "key_fixed"]
    # the entries' records are {"key", "value"} documents
    gate = next(g for g in calls_of(world, "gate") if g[2].scope)
    assert gate[1]["state"]["record"] == {"key": "math", "value": 90}
    assert gate[1]["state"]["proposed_change"]["path"] == "/value"


async def test_a_mapping_entry_is_removed_with_enable_item_removals() -> None:
    world = student_world()
    world.delete("physics", parent="Dana")
    extractor = create_extractor(
        [Many(Student)],
        transport=world.transport(),
        enable_inserts=True,
        enable_item_removals=True,
    )
    result = await extractor.ainvoke(GRADE_UPDATE, existing=[DANA])
    (record,) = result.records
    assert ("remove", "/scores/physics") in [(p.op, p.path) for p in record.patches]
    assert record.after == {"name": "Dana", "scores": {"math": 95, "chemistry": 80}}


async def test_a_mapping_of_models_changes_a_field_inside_the_entry() -> None:
    text = "Maple Mugs now has 20 mugs in stock."
    world = FakeWorld(text)
    world.entity(
        "Maple Mugs",
        schema=Shop,
        mentions=["Maple Mugs"],
        fields={"name": "Maple Mugs"},
        matches={"name": "Maple Mugs"},
    )
    world.entity(
        "mugs",
        schema="Mug",
        parent="Maple Mugs",
        mentions=["mugs"],
        fields={"key": "mugs", "value": {"quantity": 20}},
        matches={"key": "mugs"},
    )
    existing = [
        (
            "s",
            "Shop",
            {
                "name": "Maple Mugs",
                "stock": {
                    "mugs": {"quantity": 14, "colour": "blue"},
                    "plates": {"quantity": 3, "colour": None},
                },
            },
        )
    ]
    extractor = create_extractor([Many(Shop)], transport=world.transport())
    result = await extractor.ainvoke(text, existing=existing)
    (record,) = result.records
    assert ops(record) == [("replace", "/stock/mugs/quantity", 20)]
    assert record.after is not None
    assert record.after["stock"]["plates"] == {"quantity": 3, "colour": None}
    assert record.after["stock"]["mugs"] == {"quantity": 20, "colour": "blue"}


async def test_every_nested_operation_goes_through_the_change_gate() -> None:
    from trustcall_system_one.testing import clef_response, noul

    world = order_world()

    def refuse(model: str, payload: Doc, meta: Any) -> Doc:
        return clef_response(
            {qid: noul(0.1) for qid in payload["questions"]}, model=payload["model"]
        )

    extractor = create_extractor(
        [One(Order)],
        transport=FakeTransport(route(default=world.handler, gate=refuse)),
        enable_inserts=True,
    )
    result = await extractor.ainvoke(
        world.text, existing=[("SO-10442", "Order", ORDER)]
    )
    (record,) = result.records
    assert record.patches == () and record.after == record.before
    refused = {
        (r.op, r.path, r.reason) for r in record.rejected if r.reason == "gate_noul"
    }
    assert refused == {
        ("replace", "/items/1/quantity", "gate_noul"),
        ("add", "/items/-", "gate_noul"),
    }
    assert record.fields["items.1.quantity"].rejected_reason == "gate_noul"
    assert record.fields["items.3.name"].value is None


async def test_another_owners_pet_is_never_added_to_this_owners_record() -> None:
    text = "Emma Thompson walks her dog Rex. Michael Chen feeds his cat Tom."
    world = FakeWorld(text)
    for name in ("Emma Thompson", "Michael Chen"):
        world.entity(
            name,
            schema=Person,
            mentions=[name],
            fields={"name": name},
            matches={"name": name},
        )
    world.entity(
        "Rex",
        schema=Pet,
        parent="Emma Thompson",
        mentions=["Rex"],
        fields={"name": "Rex", "species": "dog"},
    )
    world.entity(
        "Tom",
        schema=Pet,
        parent="Michael Chen",
        mentions=["Tom"],
        fields={"name": "Tom", "species": "cat"},
    )
    world.bleed_item("Michael Chen", "Rex")  # offered for Michael too
    existing = [
        ("1", "Person", {"name": "Emma Thompson", "pets": []}),
        ("2", "Person", {"name": "Michael Chen", "pets": []}),
    ]
    extractor = create_extractor(
        [Many(Person)], transport=world.transport(), enable_inserts=True
    )
    result = await extractor.ainvoke(text, existing=existing)
    emma, michael = result.records
    assert ops(emma) == [("add", "/pets/-", {"name": "Rex", "species": "dog"})]
    assert ops(michael) == [("add", "/pets/-", {"name": "Tom", "species": "cat"})]
    assert [(s.reason, s.entity.label, s.entity.parent) for s in result.skipped] == [
        ("not_qualified", "Rex", "e1")
    ]


# --------------------------------------------------------------------------- #
# Item documents: TypedDicts, aliases; strictness
# --------------------------------------------------------------------------- #


def test_typed_dict_items_are_plain_dicts_in_the_parent() -> None:
    from typing_extensions import NotRequired, TypedDict

    class Toy(TypedDict):
        name: str
        colour: NotRequired[str]

    class Kid(BaseModel):
        """A kid."""

        full_name: str = Field(alias="fullName", description="name")
        toys: list[Toy] = Field(default_factory=list, description="toys")

    text = "Ada has a ball and a kite."
    world = FakeWorld(text)
    world.entity("Ada", schema=Kid, mentions=["Ada"], fields={"full_name": "Ada"})
    for toy in ("ball", "kite"):
        world.entity(
            toy, schema="Toy", parent="Ada", mentions=[toy], fields={"name": toy}
        )
    import asyncio

    extractor = create_extractor([Many(Kid)], transport=world.transport())
    result = asyncio.run(extractor.ainvoke(text))
    (record,) = result.records
    assert record.valid, record.errors
    assert record.value is not None
    assert record.value.toys == [{"name": "ball"}, {"name": "kite"}]  # type: ignore[attr-defined]


async def test_an_aliased_item_field_is_patched_under_its_alias() -> None:
    class Row(BaseModel):
        """A row."""

        name: str = Field(description="the row's name")
        count: int = Field(alias="qty", description="how many")

    class Sheet(BaseModel):
        """A sheet."""

        title: str = Field(description="the sheet's title")
        rows: Annotated[list[Row], Many(identity="name")]

    text = "On the Q3 sheet, bolts is now 7."
    world = FakeWorld(text)
    world.entity("Q3", schema=Sheet, mentions=["Q3"], fields={"title": "Q3"})
    world.entity(
        "bolts",
        schema=Row,
        parent="the Sheet",
        mentions=["bolts"],
        fields={"name": "bolts", "qty": 7, "count": 7},
        matches={"name": "bolts"},
    )
    existing = {
        "title": "Q3",
        "rows": [{"name": "nuts", "qty": 1}, {"name": "bolts", "qty": 5}],
    }
    extractor = create_extractor([One(Sheet)], transport=world.transport())
    result = await extractor.ainvoke(text, existing=[("s", "Sheet", existing)])
    (record,) = result.records
    assert ops(record) == [("replace", "/rows/1/qty", 7)]
    assert record.valid


async def test_strict_mode_raises_for_an_item_that_stays_invalid() -> None:
    from trustcall_system_one import ExtractionValidationError

    text = "Ines has 0 apples in her basket."
    world = FakeWorld(text)
    world.entity("Ines", schema=Basket, mentions=["Ines"], fields={"owner": "Ines"})
    world.entity(
        "apples",
        schema=Counted,
        parent="Ines",
        mentions=["apples"],
        fields={"name": "apples", "quantity": 0},
    )
    extractor = create_extractor(
        [Many(Basket)], transport=world.transport(), strict=True
    )
    with pytest.raises(ExtractionValidationError, match="Counted"):
        await extractor.ainvoke(text)


def test_warnings_of_an_item_name_its_ancestors() -> None:
    from trustcall_system_one import compile_schema
    from trustcall_system_one._run import Run
    from trustcall_system_one._types import EntityInfo
    from trustcall_system_one._verify import Ctx
    from trustcall_system_one.testing import FakeTransport

    run = Run(transport=FakeTransport(lambda *a: None))
    ctx = Ctx(
        run=run,
        plan=compile_schema(Pet),
        text="x",
        label="Rex",
        key="e1",
        info=EntityInfo(key="e1", label="Rex", parent="e0"),
        lineage=("Emma Thompson (e0)",),
    )
    ctx.warn("something degraded")
    sep = "\u203a"
    assert run.warnings == [f"Emma Thompson (e0) {sep} Rex (e1): something degraded"]


# --------------------------------------------------------------------------- #
# Any depth: a collection inside a sub-model
# --------------------------------------------------------------------------- #


class Household(BaseModel):
    """A household."""

    pets: list[Pet] = Field(default_factory=list, description="the pets")


class Family(BaseModel):
    """A family."""

    surname: str = Field(description="the family name")
    home: Household


async def test_a_collection_inside_a_sub_model_is_extracted_and_patched() -> None:
    text = "The Lees have Rex and Tom at home."
    world = FakeWorld(text)
    world.entity("Lee", schema=Family, mentions=["Lees"], fields={"surname": "Lee"})
    world.entity(
        "Rex",
        schema=Pet,
        parent="the Family",
        mentions=["Rex"],
        fields={"name": "Rex"},
        matches={"name": "Rex"},
    )
    world.entity(
        "Tom", schema=Pet, parent="the Family", mentions=["Tom"], fields={"name": "Tom"}
    )
    extractor = create_extractor(
        [One(Family)], transport=world.transport(), enable_inserts=True
    )
    inserted = await extractor.ainvoke(text)
    (record,) = inserted.records
    assert record.after == {
        "surname": "Lee",
        "home": {
            "pets": [
                {"name": "Rex", "species": None},
                {"name": "Tom", "species": None},
            ]
        },
    }
    assert list(record.fields) == [
        "surname",
        "home.pets.0.name",
        "home.pets.0.species",
        "home.pets.1.name",
        "home.pets.1.species",
    ]

    existing = {"surname": "Lee", "home": {"pets": [{"name": "Rex"}]}}
    updated = await extractor.ainvoke(text, existing=[("f", "Family", existing)])
    (record,) = updated.records
    assert ops(record) == [("add", "/home/pets/-", {"name": "Tom"})]
    assert record.after == {
        "surname": "Lee",
        "home": {"pets": [{"name": "Rex"}, {"name": "Tom"}]},
    }


async def test_a_list_that_must_not_be_empty_makes_the_parent_invalid_not_a_crash() -> (
    None
):
    class Squad(BaseModel):
        """A squad."""

        name: str = Field(description="the squad's name")
        members: list[Pet] = Field(min_length=1, description="the members")

    text = "The Falcons are a squad."
    world = FakeWorld(text)
    world.entity(
        "Falcons", schema=Squad, mentions=["Falcons"], fields={"name": "Falcons"}
    )
    extractor = create_extractor([Squad], transport=world.transport())
    result = await extractor.ainvoke(text)
    (record,) = result.records
    assert not record.valid
    assert record.errors and record.errors[0].startswith("members:")
    assert [s.reason for s in result.skipped] == ["validation_failed"]
    assert kinds(result)["repair"] == 0  # nothing a repair could re-ask
