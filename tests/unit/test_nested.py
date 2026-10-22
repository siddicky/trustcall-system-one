"""End-to-end tests of nested sub-models and unions of models on a fake transport."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import enum
from typing import Any, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from tests.unit.extract_helpers import (
    Script,
    ca,
    kind_counts,
    kinds,
    make_extractor,
    pick,
)
from trustcall_system_one import (
    ExtractionResult,
    ExtractionValidationError,
    GateConfig,
    Limits,
    PassesConfig,
    PresenceConfig,
    ResultRecord,
    VerifyConfig,
    s1_field,
)
from trustcall_system_one.testing import noul


class Address(BaseModel):
    """An address."""

    city: str
    zip: Optional[str] = None


class Author(BaseModel):
    name: str
    address: Optional[Address] = None


class Book(BaseModel):
    title: str
    author: Author


TEXT = (
    "The Left Hand of Darkness was written by Ursula Le Guin, "
    "who lives in Portland 97201."
)


def book_script() -> Script:
    script = Script()
    script.clairic = {
        "title": ca(
            "text", "The Left Hand of Darkness", find="The Left Hand of Darkness"
        ),
        "author.name": ca("text", "Ursula Le Guin", find="Ursula Le Guin"),
        "author.address.city": ca("text", "Portland", find="Portland"),
        "author.address.zip": ca("text", "97201", find="97201"),
    }
    return script


def extract_fields(script: Script) -> list[list[str]]:
    """The field names of every extract call, sorted, in call order."""
    return [sorted(call[1]["fields"]) for call in script.calls("extract")]


def presence_fields(script: Script) -> list[str | None]:
    return [call[2].field for call in script.calls("presence")]


async def run(
    schemas: list[Any], script: Script, text: str = TEXT, **kwargs: Any
) -> ExtractionResult:
    return await make_extractor(schemas, script, **kwargs).ainvoke(text)


def only(result: ExtractionResult) -> ResultRecord:
    (record,) = result.records
    return record


def book(record: ResultRecord) -> Book:
    assert isinstance(record.value, Book)
    return record.value


def aouter(record: ResultRecord) -> "AOuter":
    assert isinstance(record.value, AOuter)
    return record.value


def after(record: ResultRecord) -> dict[str, Any]:
    assert record.after is not None
    return record.after


def span_text(text: str, record: ResultRecord, path: str) -> str:
    span = record.fields[path].span
    assert span is not None
    return text[span[0] : span[1]]


# --------------------------------------------------------------------------- #
# Book -> Author -> Address
# --------------------------------------------------------------------------- #


async def test_book_author_address_is_extracted_end_to_end() -> None:
    script = book_script()
    result = await run([Book], script)
    record = only(result)
    assert record.valid
    assert record.value == Book(
        title="The Left Hand of Darkness",
        author=Author(
            name="Ursula Le Guin", address=Address(city="Portland", zip="97201")
        ),
    )
    assert record.after == {
        "title": "The Left Hand of Darkness",
        "author": {
            "name": "Ursula Le Guin",
            "address": {"city": "Portland", "zip": "97201"},
        },
    }
    # fields are keyed by full dotted path; spans are offsets in the original input
    assert {"title", "author.name", "author.address.city", "author.address.zip"} <= set(
        record.fields
    )
    assert span_text(TEXT, record, "title") == "The Left Hand of Darkness"
    assert span_text(TEXT, record, "author.name") == "Ursula Le Guin"
    assert span_text(TEXT, record, "author.address.city") == "Portland"
    assert span_text(TEXT, record, "author.address.zip") == "97201"
    assert record.fields["author.address.city"].value == "Portland"
    assert record.fields["author.address.city"].wire_name == "author.address.city"
    assert record.fields["author.address.city"].attribution_p == 0.95
    # one Clairic request per group
    assert sorted(extract_fields(script)) == [
        ["author.address.city", "author.address.zip"],
        ["author.name"],
        ["title"],
    ]
    assert result.warnings == ()


async def test_the_nested_call_sequence() -> None:
    script = book_script()
    await run([Book], script)
    sequence = [(meta.kind, meta.field) for _, _, meta in script.transport.calls]
    assert sequence[0] == ("presence", None)  # the entity gate first
    assert kind_counts(script.transport) == {
        "presence": 2,
        "extract": 3,
        "attribution": 1,
    }
    # the optional group is asked about before it is extracted
    group_at = sequence.index(("presence", "author.address"))
    address_at = next(
        i
        for i, (_, payload, _) in enumerate(script.transport.calls)
        if "author.address.city" in payload.get("fields", {})
    )
    assert group_at < address_at
    # the attribution check is one request over every leaf, and comes last
    (attribution,) = script.calls("attribution")
    assert sorted(attribution[1]["questions"]) == [
        "attr.author.address.city",
        "attr.author.address.zip",
        "attr.author.name",
        "attr.title",
    ]
    assert sequence[-1][0] == "attribution"


async def test_nested_instructions_carry_the_parent_context() -> None:
    script = book_script()
    await run([Book], script)
    fields = {
        name: field
        for call in script.calls("extract")
        for name, field in call[1]["fields"].items()
    }
    assert fields["title"]["instructions"] == "the Book's title"
    assert fields["author.name"]["instructions"] == "Book → author: the Book's name"
    assert (
        fields["author.address.city"]["instructions"]
        == "Book → author → address: the Book's city"
    )
    # the attribution question names the field with its ancestors
    (attribution,) = script.calls("attribution")
    question = attribution[1]["questions"]["attr.author.address.city"]
    assert question["instructions"] == (
        "Does `input` state that city (author → address) of `entity.label` "
        'is "Portland"?'
    )
    # the presence question is asked with the sub-model's own context
    (_, payload, meta) = script.calls("presence")[1]
    assert meta.field == "author.address"
    (presence_question,) = payload["questions"].values()
    assert presence_question["instructions"] == (
        "Book → author: Does `input` state the Book's address?"
    )
    assert payload["state"] == {"input": TEXT}


async def test_a_required_sub_model_is_always_asked() -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.01}
    result = await run([Book], script)
    # only the optional group got a presence question; `author` is required
    assert presence_fields(script) == [None, "author.address"]
    assert only(result).value == Book(
        title="The Left Hand of Darkness",
        author=Author(name="Ursula Le Guin"),
    )
    assert ["author.name"] in extract_fields(script)


# --------------------------------------------------------------------------- #
# Optional sub-models: absent, present, borderline
# --------------------------------------------------------------------------- #


async def test_an_absent_optional_sub_model_is_not_asked() -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.05}
    result = await run([Book], script)
    record = only(result)
    assert record.valid
    assert book(record).author.address is None
    # no Clairic request mentions the group's fields, and no attribution does
    sent = {name for fields in extract_fields(script) for name in fields}
    assert sent == {"title", "author.name"}
    asked = {
        qid for call in script.calls("attribution") for qid in call[1]["questions"]
    }
    assert asked == {"attr.title", "attr.author.name"}
    assert kind_counts(script.transport) == {
        "presence": 2,
        "extract": 2,
        "attribution": 1,
    }
    assert "author.address.city" not in record.fields
    answer = record.fields["author.address"]
    assert answer.value is None and answer.rejected_reason == "absent"
    assert answer.raw == {"noul": 0.05} and answer.source == "default"
    assert after(record)["author"] == {"name": "Ursula Le Guin", "address": None}


class Defaulted(BaseModel):
    title: str
    address: Address = Address(city="Nowhere")


class NoDefault(BaseModel):
    title: str
    address: Optional[Address]


async def test_an_absent_sub_model_gets_none_or_its_default() -> None:
    script = Script()
    script.clairic = {"title": ca("text", "Dune", find="Dune")}
    script.group_presence = {"address": 0.0}
    defaulted = only(await run([Defaulted], script, "Dune is a novel."))
    assert defaulted.value == Defaulted(title="Dune")  # the default applies
    assert defaulted.value is not None
    assert defaulted.value.address.city == "Nowhere"
    nullable = only(await run([NoDefault], script, "Dune is a novel."))
    assert nullable.value == NoDefault(title="Dune", address=None)
    assert nullable.valid


async def test_a_present_optional_sub_model_is_extracted() -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.8}
    record = only(await run([Book], script))
    assert book(record).author.address == Address(city="Portland", zip="97201")
    answer = record.fields["author.address"]
    assert answer.raw == {"noul": 0.8} and answer.source == "clef"
    assert answer.rejected_reason is None
    assert answer.value == Address(city="Portland", zip="97201")


async def test_a_borderline_sub_model_with_all_null_leaves_is_absent() -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.35}  # between 0.2 and 0.5
    script.clairic["author.address.city"] = ca("text", None, p_null=0.99)
    script.clairic["author.address.zip"] = ca("text", None, p_null=0.99)
    result = await run([Book], script)
    record = only(result)
    assert record.valid, record.errors
    assert book(record).author.address is None
    # it was asked, since the presence was not low enough to skip it
    assert ["author.address.city", "author.address.zip"] in extract_fields(script)
    answer = record.fields["author.address"]
    assert answer.rejected_reason == "absent" and answer.raw == {"noul": 0.35}


async def test_a_borderline_sub_model_with_a_value_is_kept() -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.35}
    script.clairic["author.address.city"] = ca("text", None, p_null=0.99)
    record = only(await run([Book], script))
    # only the zip came back, but that is evidence: the sub-model stands, and
    # its missing required field is reported by validation
    assert not record.valid
    assert record.errors == ("author.address.city: Input should be a valid string",)
    assert after(record)["author"]["address"] == {"city": None, "zip": "97201"}


async def test_the_presence_thresholds_are_configurable() -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.35}
    script.clairic["author.address.city"] = ca("text", None, p_null=0.99)
    script.clairic["author.address.zip"] = ca("text", None, p_null=0.99)
    result = await run(
        [Book],
        script,
        presence=PresenceConfig(threshold_absent=0.4, threshold_present=0.5),
    )
    assert only(result).value is not None
    assert only(result).value.author.address is None  # type: ignore[union-attr]
    assert ["author.address.city", "author.address.zip"] not in extract_fields(script)


async def test_a_confident_sub_model_with_null_leaves_is_not_made_absent() -> None:
    script = book_script()  # presence 0.95
    script.clairic["author.address.city"] = ca("text", None, p_null=0.99)
    script.clairic["author.address.zip"] = ca("text", None, p_null=0.99)
    record = only(await run([Book], script))
    assert not record.valid  # present but its required city is missing
    assert record.errors[0].startswith("author.address.city")


class Deep(BaseModel):
    title: str
    author: Optional[Author] = None


async def test_nested_presence_is_asked_only_below_a_present_parent() -> None:
    script = book_script()
    script.group_presence = {"author": 0.0, "author.address": 0.99}
    result = await run([Deep], script)
    assert only(result).value == Deep(title="The Left Hand of Darkness")
    # the grand-child's presence was never asked, because its parent is absent
    assert presence_fields(script) == [None, "author"]
    assert extract_fields(script) == [["title"]]
    script = book_script()
    script.group_presence = {"author": 0.9, "author.address": 0.1}
    result = await run([Deep], script)
    assert only(result).value == Deep(
        title="The Left Hand of Darkness", author=Author(name="Ursula Le Guin")
    )
    assert presence_fields(script).count("author.address") == 1


# --------------------------------------------------------------------------- #
# passes.split
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("split", "expected"),
    [("entity", 1), ("submodel", 3), ("field", 4)],
)
async def test_passes_split_sets_the_number_of_clairic_requests(
    split: str, expected: int
) -> None:
    script = book_script()
    result = await run([Book], script, passes=PassesConfig(split=split))  # type: ignore[arg-type]
    record = only(result)
    assert record.valid
    assert book(record).author.address == Address(city="Portland", zip="97201")
    assert len(script.calls("extract")) == expected
    assert span_text(TEXT, record, "author.address.zip") == "97201"
    if split == "entity":
        assert extract_fields(script) == [
            [
                "author.address.city",
                "author.address.zip",
                "author.name",
                "title",
            ]
        ]
    if split == "field":
        assert sorted(f for fields in extract_fields(script) for f in fields) == [
            "author.address.city",
            "author.address.zip",
            "author.name",
            "title",
        ]
        assert all(len(fields) == 1 for fields in extract_fields(script))


@pytest.mark.parametrize(
    ("split", "expected"), [("entity", 1), ("submodel", 2), ("field", 2)]
)
async def test_an_absent_group_is_not_asked_under_any_split_but_entity(
    split: str, expected: int
) -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.0}
    result = await run([Book], script, passes=PassesConfig(split=split))  # type: ignore[arg-type]
    record = only(result)
    assert record.valid
    assert book(record).author.address is None
    # entity: one request for everything, and the absent group's answers are
    # discarded; the others skip the group's request altogether
    assert len(script.calls("extract")) == expected
    assert "author.address.city" not in record.fields


# --------------------------------------------------------------------------- #
# Verification inside nested fields
# --------------------------------------------------------------------------- #


async def test_attribution_drops_a_nested_value() -> None:
    script = book_script()
    script.attribution = {"attr.author.address.zip": 0.05}
    record = only(await run([Book], script))
    assert record.valid
    assert book(record).author.address == Address(city="Portland", zip=None)
    answer = record.fields["author.address.zip"]
    assert answer.rejected_reason == "attribution" and answer.attribution_p == 0.05
    assert answer.value is None
    (rejected,) = record.rejected
    assert (rejected.path, rejected.reason) == ("/author/address/zip", "attribution")


async def test_a_nested_low_confidence_value_is_asked_again() -> None:
    script = book_script()
    script.clairic["author.address.city"] = ca(
        "text", "Portland", find="Portland", confidence=0.6
    )
    script.reask = {"author.address.city": ca("text", "Portland", find="Portland")}
    record = only(await run([Book], script))
    assert book(record).author.address == Address(city="Portland", zip="97201")
    assert record.fields["author.address.city"].reask is True
    (call,) = script.calls("reask")
    assert call[2].field == "author.address.city"
    assert list(call[1]["fields"]) == ["author.address.city"]
    assert call[1]["fields"]["author.address.city"]["instructions"].startswith(
        "Book → author → address: the Book's city"
    )


async def test_a_nested_borderline_null_is_checked_with_its_ancestors_named() -> None:
    script = book_script()
    script.clairic["author.address.zip"] = ca("text", None, p_null=0.5)
    script.null_check = {"nullcheck.author.address.zip": 0.1}
    record = only(await run([Book], script))
    assert book(record).author.address == Address(city="Portland", zip=None)
    (call,) = script.calls("null_check")
    question = call[1]["questions"]["nullcheck.author.address.zip"]
    assert question["instructions"] == (
        "Does `input` state the Book's zip (author → address)?"
    )


async def test_nested_disagreement_goes_to_adjudication() -> None:
    script = book_script()
    script.clairic["author.name"] = ca(
        "text", "Ursula Le Guin", find="Ursula Le Guin", confidence=0.6
    )
    script.reask = {"author.name": ca("text", "Le Guin", find="Le Guin")}
    script.adjudicate = {"author.name": pick("a", ["a", "b", "neither"])}
    record = only(await run([Book], script))
    assert book(record).author.name == "Ursula Le Guin"
    (call,) = script.calls("adjudicate")
    assert call[2].field == "author.name"
    question = call[1]["questions"]["adjudicate"]
    assert "name (author)" in question["instructions"]


# --------------------------------------------------------------------------- #
# Typed leaves at any depth
# --------------------------------------------------------------------------- #


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Detail(BaseModel):
    color: Color
    happy: bool
    tags: list[Literal["a", "b"]] = []
    size: Union[int, str, None] = None
    kind: Literal["detail"] = "detail"


class Middle(BaseModel):
    detail: Detail


class Holder(BaseModel):
    name: str
    middle: Middle


async def test_clef_typed_leaves_and_unions_work_at_any_depth() -> None:
    script = Script()
    script.clairic = {
        "name": ca("text", "Box", find="Box"),
        "middle.detail.tags": ca("list", ["a b"], find_all=["a b"]),
        "middle.detail.size": ca("text", "12", find="12"),
    }
    script.clef = {
        "middle.detail.color": pick("BLUE", ["RED", "BLUE"]),
        "middle.detail.happy": noul(0.9),
        "middle.detail.size.__type": pick("int", ["int", "str"]),
        "middle.detail.tags.el0": pick("a", ["a", "b", "none_of_these"]),
    }
    text = "Box: a blue happy thing, size 12, labelled a b."
    result = await run([Holder], script, text)
    record = only(result)
    assert record.valid, record.errors
    assert record.value == Holder(
        name="Box",
        middle=Middle(detail=Detail(color=Color.BLUE, happy=True, tags=["a"], size=12)),
    )
    assert record.fields["middle.detail.color"].value == Color.BLUE
    assert record.fields["middle.detail.happy"].value is True
    assert record.fields["middle.detail.tags"].value == ["a"]
    assert record.fields["middle.detail.size"].value == 12
    assert record.fields["middle.detail.size"].raw == {"text": "12", "type": "int"}
    assert span_text(text, record, "middle.detail.size") == "12"
    # the Clef questions of the nested leaves go with the group's request
    assert any(
        "middle.detail.color" in call[1]["questions"]
        for call in script.calls("clef_fields")
    )


# --------------------------------------------------------------------------- #
# Aliases and pruning, end to end
# --------------------------------------------------------------------------- #


class AInner(BaseModel):
    val: int = Field(alias="v")
    note: Optional[str] = Field(None, alias="n")


class AOuter(BaseModel):
    inner: AInner = Field(alias="in")
    top: Optional[str] = Field(None, alias="t")


async def test_aliases_work_at_two_levels() -> None:
    script = Script()
    script.clairic = {
        "inner.val": ca("number", 7),
        "inner.note": ca("text", None, p_null=0.99),
        "top": ca("text", "hi", find="hi"),
    }
    record = only(await run([AOuter], script, "hi, 7"))
    assert record.valid, record.errors
    assert aouter(record).inner.val == 7 and aouter(record).inner.note is None
    assert aouter(record).top == "hi"
    # None is pruned at the nested level (the default applies), and the document
    # handed to validation is keyed by the aliases
    assert record.fields["inner.val"].value == 7
    assert record.fields["inner.note"].value is None
    assert record.fields["inner.note"].source == "default"


# --------------------------------------------------------------------------- #
# Required sub-model gaps and strict mode
# --------------------------------------------------------------------------- #


async def test_a_missing_required_nested_value_makes_the_record_invalid() -> None:
    script = book_script()
    script.clairic["author.name"] = ca("text", None, p_null=0.99)
    record = only(await run([Book], script))
    assert record.valid is False
    assert record.value is None
    assert record.errors[0].startswith("author.name:")
    assert after(record)["author"]["name"] is None
    # the answers are still reported by full dotted path
    assert record.fields["author.address.city"].value == "Portland"


async def test_strict_raises_for_an_invalid_nested_record() -> None:
    script = book_script()
    script.clairic["author.name"] = ca("text", None, p_null=0.99)
    with pytest.raises(ExtractionValidationError, match=r"author\.name"):
        await make_extractor([Book], script, strict=True).ainvoke(TEXT)


class Node(BaseModel):
    name: str
    parent: Optional["Node"] = None


Node.model_rebuild()


async def test_a_self_recursive_optional_model_unrolls_with_presence_questions() -> (
    None
):
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada", find="Ada"),
        "parent.name": ca("text", "Byron", find="Byron"),
        "parent.parent.name": ca("text", "Anne", find="Anne"),
    }
    script.group_presence = {"parent": 0.9, "parent.parent": 0.05}
    result = await run(
        [Node], script, "Ada, child of Byron.", limits=Limits(max_depth=4)
    )
    record = only(result)
    assert record.value == Node(name="Ada", parent=Node(name="Byron"))
    assert presence_fields(script) == [None, "parent", "parent.parent"]
    assert record.fields["parent.parent"].rejected_reason == "absent"


# --------------------------------------------------------------------------- #
# Unions of models
# --------------------------------------------------------------------------- #


class Cat(BaseModel):
    """A house cat."""

    kind: Literal["cat"]
    lives: int


class Dog(BaseModel):
    kind: Literal["dog"]
    tricks: int
    nickname: Optional[str] = None


class Owner(BaseModel):
    name: str
    pet: Union[Cat, Dog]


OWNER_TEXT = "Sam owns a dog called Rex who knows 3 tricks."


def owner_script(choice: str = "dog", p: float = 0.95) -> Script:
    script = Script()
    script.clairic = {
        "name": ca("text", "Sam", find="Sam"),
        "pet.Cat.lives": ca("number", 9),
        "pet.Dog.tricks": ca("number", 3),
        "pet.Dog.nickname": ca("text", "Rex", find="Rex"),
    }
    script.clef = {"pet": pick(choice, ["cat", "dog"], p)}
    return script


async def test_a_union_asks_only_the_chosen_members_fields() -> None:
    script = owner_script()
    result = await run([Owner], script, OWNER_TEXT)
    record = only(result)
    assert record.valid, record.errors
    assert record.value == Owner(
        name="Sam", pet=Dog(kind="dog", tricks=3, nickname="Rex")
    )
    sent = {name for fields in extract_fields(script) for name in fields}
    assert sent == {"name", "pet.Dog.tricks", "pet.Dog.nickname"}
    assert sorted(kinds(script.transport)) == [
        "attribution",
        "clef_fields",
        "extract",
        "extract",
        "presence",
    ]
    # the choice goes out with the root pass, before the member is extracted
    order = kinds(script.transport)
    assert order.index("clef_fields") < len(order) - 1
    (clef_call,) = script.calls("clef_fields")
    assert list(clef_call[1]["questions"]) == ["pet"]
    question = clef_call[1]["questions"]["pet"]
    assert question["instructions"] == "Pick the option that best fits: the Owner's pet"
    assert question["criteria"]["cat"] == "A house cat."
    # the discriminator is filled in from the chosen member
    assert record.fields["pet.kind"].value == "dog"
    assert record.fields["pet.kind"].source == "const"
    assert record.fields["pet.tricks"].value == 3
    assert span_text(OWNER_TEXT, record, "pet.nickname") == "Rex"
    answer = record.fields["pet"]
    assert answer.engine == "clef" and answer.source == "clef"
    assert isinstance(answer.value, Dog)
    assert answer.raw["choice"] == "dog"
    # only the chosen member's answers are in the result
    assert "pet.lives" not in record.fields
    # attribution covers the member's leaves only
    asked = {
        qid for call in script.calls("attribution") for qid in call[1]["questions"]
    }
    assert asked == {"attr.name", "attr.pet.Dog.tricks", "attr.pet.Dog.nickname"}


async def test_the_other_member_is_chosen_by_the_choice() -> None:
    script = owner_script("cat")
    record = only(await run([Owner], script, "Sam owns a cat with 9 lives."))
    assert record.value == Owner(name="Sam", pet=Cat(kind="cat", lives=9))
    sent = {name for fields in extract_fields(script) for name in fields}
    assert sent == {"name", "pet.Cat.lives"}


async def test_the_union_choice_is_asked_with_the_root_pass() -> None:
    script = owner_script()
    await run([Owner], script, OWNER_TEXT)
    sequence = [(meta.kind, meta.field) for _, _, meta in script.transport.calls]
    # the root's extract call, its clef_fields call and the choice are issued
    # together; the member pass follows the choice
    clef_at = sequence.index(("clef_fields", None))
    member_at = next(
        i
        for i, (_, payload, meta) in enumerate(script.transport.calls)
        if meta.kind == "extract" and "pet.Dog.tricks" in payload["fields"]
    )
    assert clef_at < member_at


class MaybeOwner(BaseModel):
    name: str
    pet: Optional[Union[Cat, Dog]] = None


async def test_an_optional_union_can_be_unstated() -> None:
    script = owner_script()
    script.clef = {"pet": pick("unstated", ["cat", "dog", "unstated"])}
    record = only(await run([MaybeOwner], script, "Sam says hello."))
    assert record.valid
    assert record.value == MaybeOwner(name="Sam")
    sent = {name for fields in extract_fields(script) for name in fields}
    assert sent == {"name"}  # no member was asked
    assert record.fields["pet"].value is None
    assert record.fields["pet"].source == "default"


async def test_an_optional_union_with_a_confident_choice_is_extracted() -> None:
    script = owner_script()
    script.clef = {"pet": pick("dog", ["cat", "dog", "unstated"])}
    record = only(await run([MaybeOwner], script, OWNER_TEXT))
    assert record.value == MaybeOwner(
        name="Sam", pet=Dog(kind="dog", tricks=3, nickname="Rex")
    )


async def test_a_low_confidence_choice_on_an_optional_union_is_unstated() -> None:
    script = owner_script()
    script.clef = {"pet": pick("dog", ["cat", "dog", "unstated"], 0.4)}
    result = await run([MaybeOwner], script, OWNER_TEXT)
    record = only(result)
    assert record.value == MaybeOwner(name="Sam")
    assert any("not confident" in w and "unstated" in w for w in result.warnings)
    assert {name for fields in extract_fields(script) for name in fields} == {"name"}
    assert [(r.path, r.reason) for r in record.rejected] == [
        ("/pet", "low_choice_prob")
    ]


async def test_a_low_confidence_choice_on_a_required_union_extracts_every_member() -> (
    None
):
    script = owner_script("dog", 0.5)  # below min_choice_prob (0.55)
    result = await run([Owner], script, OWNER_TEXT)
    record = only(result)
    # the member with the most verified non-null fields wins: Dog has two
    assert record.value == Owner(
        name="Sam", pet=Dog(kind="dog", tricks=3, nickname="Rex")
    )
    sent = {name for fields in extract_fields(script) for name in fields}
    assert sent == {"name", "pet.Cat.lives", "pet.Dog.tricks", "pet.Dog.nickname"}
    assert any(
        "every member was extracted" in w and "Owner.pet" in w for w in result.warnings
    )
    assert record.fields["pet"].rejected_reason == "low_confidence_fallback"
    assert "pet.lives" not in record.fields and "pet.tricks" in record.fields


async def test_the_fallback_prefers_verified_values() -> None:
    script = owner_script("dog", 0.5)
    # Dog's values do not survive attribution; Cat's one does
    script.attribution = {
        "attr.pet.Dog.tricks": 0.01,
        "attr.pet.Dog.nickname": 0.01,
    }
    record = only(await run([Owner], script, "Sam owns something with 9 lives."))
    assert record.value == Owner(name="Sam", pet=Cat(kind="cat", lives=9))


async def test_a_fallback_with_nothing_found_leaves_the_union_unset() -> None:
    script = owner_script("dog", 0.5)
    script.clairic["pet.Cat.lives"] = ca("number", None, p_null=0.99)
    script.clairic["pet.Dog.tricks"] = ca("number", None, p_null=0.99)
    script.clairic["pet.Dog.nickname"] = ca("text", None, p_null=0.99)
    record = only(await run([Owner], script, "Sam says hello."))
    assert not record.valid
    assert record.errors[0].startswith("pet:")


async def test_an_unsure_choice_confidence_makes_the_union_fall_back() -> None:
    script = owner_script()
    script.clef = {
        "pet": {
            "type": "choice",
            "choice": "dog",
            "probabilities": {"cat": 0.3, "dog": 0.7},
            "confidence": 0.3,  # below min_choice_confidence (0.5)
        }
    }
    result = await run([Owner], script, OWNER_TEXT)
    assert only(result).valid
    assert any("every member was extracted" in w for w in result.warnings)
    assert [(r.path, r.reason) for r in only(result).rejected] == [
        ("/pet", "low_confidence")
    ]


class Big(BaseModel):
    size: Literal["large", "huge"]
    weight: int


class Small(BaseModel):
    size: Literal["small"]
    length: int


class Sized(BaseModel):
    item: Union[Big, Small]


async def test_a_tag_with_several_values_is_filled_from_the_choice() -> None:
    script = Script()
    script.clairic = {"item.Big.weight": ca("number", 90)}
    script.clef = {"item": pick("huge", ["large", "huge", "small"])}
    record = only(await run([Sized], script, "A huge crate weighing 90 kg."))
    assert record.value == Sized(item=Big(size="huge", weight=90))
    assert record.fields["item.size"].value == "huge"
    # the tag was not asked as a question of its own
    (clef_call,) = script.calls("clef_fields")
    assert list(clef_call[1]["questions"]) == ["item"]


class Wing(BaseModel):
    span: float
    tip: Optional[Address] = None


class Eagle(BaseModel):
    kind: Literal["eagle"]
    wing: Wing


class Owl(BaseModel):
    kind: Literal["owl"]
    hoots: int


class Sky(BaseModel):
    bird: Union[Eagle, Owl]


async def test_union_members_may_contain_optional_sub_models() -> None:
    script = Script()
    script.clairic = {
        "bird.Eagle.wing.span": ca("number", 2.1),
        "bird.Eagle.wing.tip.city": ca("text", "Reno", find="Reno"),
        "bird.Eagle.wing.tip.zip": ca("text", None, p_null=0.99),
    }
    script.clef = {"bird": pick("eagle", ["eagle", "owl"])}
    script.group_presence = {"bird.wing.tip": 0.9}
    text = "An eagle with a 2.1 m wingspan, tagged in Reno."
    record = only(await run([Sky], script, text))
    assert record.value == Sky(
        bird=Eagle(kind="eagle", wing=Wing(span=2.1, tip=Address(city="Reno")))
    )
    assert presence_fields(script) == [None, "bird.wing.tip"]


class Zoo(BaseModel):
    name: str
    keeper: Optional["Keeper"] = None


class Keeper(BaseModel):
    pet: Union[Cat, Dog]


Zoo.model_rebuild()


async def test_a_union_below_an_absent_sub_model_is_never_asked() -> None:
    script = owner_script()
    script.group_presence = {"keeper": 0.0}
    record = only(await run([Zoo], script, "Sam runs a zoo."))
    assert record.value == Zoo(name="Sam")
    assert script.calls("clef_fields") == []


async def test_a_union_member_is_keyed_by_the_aliases() -> None:
    class Pup(BaseModel):
        kind: Literal["pup"]
        age: int = Field(alias="a")

    class Kit(BaseModel):
        kind: Literal["kit"]
        age: int = Field(alias="a")

    class Home(BaseModel):
        pet: Union[Pup, Kit] = Field(alias="p")

    script = Script()
    script.clairic = {"pet.Pup.age": ca("number", 2), "pet.Kit.age": ca("number", 1)}
    script.clef = {"pet": pick("pup", ["pup", "kit"])}
    record = only(await run([Home], script, "A pup aged 2."))
    assert record.value == Home.model_validate({"p": {"kind": "pup", "a": 2}})
    assert record.after == {"pet": {"kind": "pup", "age": 2}}


# --------------------------------------------------------------------------- #
# Truncation, per group pass
# --------------------------------------------------------------------------- #


def filler(count: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(count)) + " "


async def test_nested_groups_are_chunked_on_their_own() -> None:
    text = (
        "Title: Dune. "
        + filler(1500, "a")
        + "Written by Frank Herbert. "
        + filler(1500, "b")
        + "He lived in Tacoma 98402. "
        + filler(1500, "c")
    )
    script = Script()
    script.clairic = {
        "title": ca("text", "Dune", find="Dune"),
        "author.name": ca("text", "Frank Herbert", find="Frank Herbert"),
        "author.address.city": ca("text", "Tacoma", find="Tacoma"),
        "author.address.zip": ca("text", "98402", find="98402"),
    }
    # any request over 6000 characters is cut short and flagged truncated
    script.limit = lambda state, n: 6000 if len(state) > 6000 else None
    result = await run([Book], script, text, verify=VerifyConfig(window_chars=100))
    record = only(result)
    assert record.valid, record.errors
    assert record.value == Book(
        title="Dune",
        author=Author(
            name="Frank Herbert", address=Address(city="Tacoma", zip="98402")
        ),
    )
    # every group was chunked separately: chunk requests only ever carry the
    # fields of their own group
    groups = {
        ("title",): 0,
        ("author.name",): 0,
        ("author.address.city", "author.address.zip"): 0,
    }
    for fields in extract_fields(script):
        groups[tuple(fields)] += 1
    assert all(count >= 3 for count in groups.values()), groups  # full + chunks
    # spans are mapped back to the original input
    assert span_text(text, record, "title") == "Dune"
    assert span_text(text, record, "author.name") == "Frank Herbert"
    assert span_text(text, record, "author.address.city") == "Tacoma"
    assert span_text(text, record, "author.address.zip") == "98402"
    assert any("clairic_chunked" in w for w in result.warnings)
    assert not any("clairic_truncated" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


async def test_forcing_the_schema_skips_the_entity_presence_gate_only() -> None:
    script = book_script()
    result = await make_extractor([Book], script).ainvoke(TEXT, schema=Book)
    assert presence_fields(script) == ["author.address"]  # the group's gate stays
    assert only(result).valid


async def test_flat_schemas_are_unchanged() -> None:
    class Flat(BaseModel):
        title: str

    script = Script()
    script.clairic = {"title": ca("text", "Dune", find="Dune")}
    result = await run([Flat], script, "Dune")
    assert only(result).value == Flat(title="Dune")
    assert sorted(kinds(script.transport)) == ["attribution", "extract", "presence"]


async def test_the_choice_threshold_comes_from_the_gate_config() -> None:
    script = owner_script("dog", 0.7)
    result = await run(
        [Owner], script, OWNER_TEXT, gates=GateConfig(min_choice_prob=0.8)
    )
    assert any("every member was extracted" in w for w in result.warnings)


async def test_s1_field_skip_and_instructions_apply_to_nested_fields() -> None:
    class Inner(BaseModel):
        a: str = Field(json_schema_extra=s1_field(instructions="the first thing"))
        b: str = Field("x", json_schema_extra=s1_field(skip=True))

    class Outer(BaseModel):
        inner: Inner

    script = Script()
    script.clairic = {"inner.a": ca("text", "foo", find="foo")}
    record = only(await run([Outer], script, "foo"))
    assert record.value == Outer(inner=Inner(a="foo", b="x"))
    (call,) = script.calls("extract")
    assert list(call[1]["fields"]) == ["inner.a"]
    assert (
        call[1]["fields"]["inner.a"]["instructions"]
        == "Outer → inner: the Outer's the first thing"
    )


@pytest.mark.parametrize(
    ("split", "expected"), [("entity", 2), ("submodel", 2), ("field", 3)]
)
async def test_a_union_member_is_one_unit_of_passes_under_every_split(
    split: str, expected: int
) -> None:
    script = owner_script()
    result = await run(
        [Owner],
        script,
        OWNER_TEXT,
        passes=PassesConfig(split=split),  # type: ignore[arg-type]
    )
    record = only(result)
    assert record.valid, record.errors
    assert record.value == Owner(
        name="Sam", pet=Dog(kind="dog", tricks=3, nickname="Rex")
    )
    # the root, then the chosen member (two fields: one pass or one per field)
    assert len(script.calls("extract")) == expected
