"""End-to-end tests of nested sub-models and unions of models on a fake transport."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import enum
from typing import Any, Literal, Optional, Union, cast

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
    ModelResponseError,
    PassesConfig,
    PresenceConfig,
    ResultRecord,
    TransportError,
    TrustcallSystemOneError,
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
    # nested fields are named through their owners: the author's name, not the
    # Book's (which would invite the title)
    assert (
        fields["author.name"]["instructions"]
        == "Book → author: the Book's author's name"
    )
    assert (
        fields["author.address.city"]["instructions"]
        == "Book → author → address: the Book's author's address's city"
    )
    # the attribution question names the field with its ancestors
    (attribution,) = script.calls("attribution")
    question = attribution[1]["questions"]["attr.author.address.city"]
    assert question["instructions"] == (
        "Does `input` state or directly imply (including simple arithmetic from "
        "stated figures, or date math) that city (author → address) of "
        '`entity.label` is "Portland"?'
    )
    # the presence question is asked with the sub-model's own context
    (_, payload, meta) = script.calls("presence")[1]
    assert meta.field == "author.address"
    (presence_question,) = payload["questions"].values()
    assert presence_question["instructions"] == (
        "Book → author: Does `input` state the Book's author's address?"
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


async def test_a_borderline_sub_model_with_a_value_but_a_gap_is_dropped() -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.35}
    script.clairic["author.address.city"] = ca("text", None, p_null=0.99)
    result = await run([Book], script)
    record = only(result)
    # only the zip came back: the sub-model stands as evidence, but it cannot be
    # built without its required city, so it is dropped and the record survives
    assert record.valid, record.errors
    assert book(record).author.address is None
    (rejected,) = [r for r in record.rejected if r.reason == "invalid_optional_group"]
    assert rejected.path == "/author/address" and rejected.op is None
    assert "author.address.city" in rejected.detail
    assert any("treated as absent" in w for w in result.warnings)
    answer = record.fields["author.address"]
    assert answer.value is None and answer.rejected_reason == "invalid_optional_group"
    # the leaf answers it held are reported as dropped, not as standing
    zip_answer = record.fields["author.address.zip"]
    assert zip_answer.value is None
    assert zip_answer.rejected_reason == "invalid_optional_group"


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


async def test_a_confident_sub_model_with_null_leaves_is_dropped_if_invalid() -> None:
    script = book_script()  # presence 0.95: not borderline, so not "all null"
    script.clairic["author.address.city"] = ca("text", None, p_null=0.99)
    script.clairic["author.address.zip"] = ca("text", None, p_null=0.99)
    result = await run([Book], script)
    record = only(result)
    assert record.valid, record.errors
    assert book(record).author.address is None
    (rejected,) = [r for r in record.rejected if r.reason == "invalid_optional_group"]
    assert rejected.path == "/author/address"
    assert record.fields["author.address"].rejected_reason == "invalid_optional_group"
    assert any("author.address" in w and "absent" in w for w in result.warnings)


async def test_an_invalid_required_sub_model_still_invalidates_the_record() -> None:
    script = book_script()
    script.clairic["author.name"] = ca("text", None, p_null=0.99)
    record = only(await run([Book], script))
    assert not record.valid
    assert record.errors[0].startswith("author.name:")


async def test_a_group_below_a_dropped_one_goes_with_it() -> None:
    class Inner(BaseModel):
        code: str
        extra: Optional[Address] = None

    class Mid(BaseModel):
        title: str
        inner: Optional[Inner] = None

    script = Script()
    script.clairic = {
        "title": ca("text", "Dune", find="Dune"),
        "inner.code": ca("text", None, p_null=0.99),
        "inner.extra.city": ca("text", "Reno", find="Reno"),
        "inner.extra.zip": ca("text", None, p_null=0.99),
    }
    record = only(await run([Mid], script, "Dune, Reno"))
    # extra stands (valid), inner fails on its code: inner and what is in it go
    assert record.valid
    assert record.value == Mid(title="Dune")
    paths = [r.path for r in record.rejected if r.reason == "invalid_optional_group"]
    assert paths == ["/inner"]
    assert record.fields["inner.extra.city"].value is None
    # the notes of what sits below the dropped group go with it as well: no
    # grandchild keeps saying "clef" (present) or has no reason
    for path in ("inner", "inner.extra"):
        note = record.fields[path]
        assert note.value is None and note.source == "default", path
        assert note.rejected_reason == "invalid_optional_group", path


async def test_a_union_below_a_dropped_group_goes_with_it() -> None:
    class Cat(BaseModel):
        kind: Literal["cat"] = "cat"
        lives: int

    class Dog(BaseModel):
        kind: Literal["dog"] = "dog"
        good: bool

    class Owner(BaseModel):
        name: str
        pet: Union[Cat, Dog]

    class Home(BaseModel):
        title: str
        owner: Optional[Owner] = None

    script = Script()
    script.clairic = {
        "title": ca("text", "Cosy", find="Cosy"),
        "owner.name": ca("text", None, p_null=0.99),  # required: the group fails
        "owner.pet.Cat.lives": ca("number", 9, find="9"),
    }
    script.clef = {"owner.pet": pick("Cat", ["Cat", "Dog"])}
    record = only(await run([Home], script, "Cosy, 9"))
    assert record.valid and record.value == Home(title="Cosy")
    assert record.fields["owner"].rejected_reason == "invalid_optional_group"
    choice = record.fields["owner.pet"]
    assert choice.value is None and choice.source == "default"
    assert choice.rejected_reason == "invalid_optional_group"


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
        "Book → author → address: the Book's author's address's city"
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
        "Does `input` state the Book's author's address's zip?"
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
    assert answer.engine == "union_models" and answer.source == "clef"
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
        == "Outer → inner: the Outer's inner's first thing"
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


# --------------------------------------------------------------------------- #
# Review fixes: union members are validated as themselves
# --------------------------------------------------------------------------- #


class Kitty(BaseModel):
    """A cat."""

    name: str
    indoor: bool = False


class Pup(BaseModel):
    """A dog."""

    name: str
    breed: Optional[str] = None


class Household(BaseModel):
    owner: str
    pet: Union[Kitty, Pup]


def household_script(choice: str, p: float = 0.95) -> Script:
    script = Script()
    script.clairic = {
        "owner": ca("text", "Sam", find="Sam"),
        "pet.Pup.name": ca("text", "Rex", find="Rex"),
        "pet.Pup.breed": ca("text", None, p_null=0.99),
        "pet.Kitty.name": ca("text", "Rex", find="Rex"),
    }
    script.clef = {
        "pet": pick(choice, ["Kitty", "Pup"], p),
        "pet.Kitty.indoor": noul(0.05),
    }
    return script


async def test_the_chosen_member_survives_validation_when_members_overlap() -> None:
    # Pup's document {"name": "Rex"} also validates as a Kitty: pydantic's smart
    # union would pick the Kitty. The chosen member is validated as itself.
    script = household_script("Pup")
    record = only(await run([Household], script, "Sam has a dog named Rex."))
    assert record.valid, record.errors
    assert record.value == Household(owner="Sam", pet=Pup(name="Rex"))
    assert type(record.value.pet) is Pup
    assert after(record)["pet"] == {"name": "Rex", "breed": None}
    answer = record.fields["pet"]
    assert answer.engine == "union_models" and isinstance(answer.value, Pup)


async def test_the_other_overlapping_member_can_be_chosen_too() -> None:
    script = household_script("Kitty")
    script.clef["pet.Kitty.indoor"] = noul(0.9)
    record = only(await run([Household], script, "Sam has an indoor cat named Rex."))
    assert record.value == Household(owner="Sam", pet=Kitty(name="Rex", indoor=True))
    assert type(record.value.pet) is Kitty


async def test_an_invalid_chosen_member_is_reported_not_swapped_for_another() -> None:
    class Cat(BaseModel):
        name: str
        lives: Optional[int] = None

    class Dog(BaseModel):
        name: str
        tricks: int

    class Home(BaseModel):
        pet: Union[Cat, Dog]

    script = Script()
    script.clairic = {
        "pet.Dog.name": ca("text", "Rex", find="Rex"),
        "pet.Dog.tricks": ca("number", None, p_null=0.99),
        "pet.Cat.name": ca("text", "Rex", find="Rex"),
        "pet.Cat.lives": ca("number", None, p_null=0.99),
    }
    script.clef = {"pet": pick("Dog", ["Cat", "Dog"])}
    result = await run([Home], script, "A dog named Rex.")
    record = only(result)
    # {"name": "Rex"} would be a valid Cat, but the choice was Dog
    assert not record.valid and record.value is None
    assert record.errors == ("pet.tricks: Input should be a valid integer",)
    assert record.fields["pet"].value is None
    assert record.fields["pet"].raw["choice"] == "Dog"


async def test_an_invalid_member_of_an_optional_union_is_dropped() -> None:
    class Dog(BaseModel):
        name: str
        tricks: int

    class Fish(BaseModel):
        water: str

    class Home(BaseModel):
        owner: str
        pet: Optional[Union[Dog, Fish]] = None

    script = Script()
    script.clairic = {
        "owner": ca("text", "Sam", find="Sam"),
        "pet.Dog.name": ca("text", "Rex", find="Rex"),
        "pet.Dog.tricks": ca("number", None, p_null=0.99),
    }
    script.clef = {"pet": pick("Dog", ["Dog", "Fish", "unstated"])}
    result = await run([Home], script, "Sam has a dog named Rex.")
    record = only(result)
    assert record.valid and book_free(record) == Home(owner="Sam")
    assert [
        r.path for r in record.rejected if r.reason == "invalid_optional_group"
    ] == ["/pet"]
    assert record.fields["pet"].rejected_reason == "invalid_optional_group"
    assert any("failed validation" in w for w in result.warnings)


def book_free(record: ResultRecord) -> BaseModel:
    assert record.value is not None
    return record.value


async def test_the_fallback_member_is_validated_as_itself_too() -> None:
    script = household_script("Pup", 0.5)
    result = await run([Household], script, "Sam has a dog named Rex.")
    record = only(result)
    assert record.valid, record.errors
    # Kitty and Pup each found one value: the tie goes to the earlier member
    assert type(record.value.pet) is Kitty  # type: ignore[union-attr]


async def test_discriminated_unions_still_validate_through_instances() -> None:
    script = owner_script()
    record = only(await run([Owner], script, OWNER_TEXT))
    assert type(record.value.pet) is Dog  # type: ignore[union-attr]
    assert record.valid


async def test_a_typed_dict_sub_model_stays_a_dict() -> None:
    from typing_extensions import TypedDict

    class Geo(TypedDict):
        lat: float
        lon: float

    class Place(BaseModel):
        name: str
        geo: Optional[Geo] = None

    script = Script()
    script.clairic = {
        "name": ca("text", "Oslo", find="Oslo"),
        "geo.lat": ca("number", 59.9),
        "geo.lon": ca("number", 10.7),
    }
    record = only(await run([Place], script, "Oslo at 59.9, 10.7."))
    assert record.valid, record.errors
    assert record.value == Place(name="Oslo", geo={"lat": 59.9, "lon": 10.7})


# --------------------------------------------------------------------------- #
# Review fixes: only Clairic values are evidence
# --------------------------------------------------------------------------- #


class Checked(BaseModel):
    city: str
    verified: bool = False


class Person(BaseModel):
    name: str
    address: Optional[Checked] = None


async def test_a_bool_leaf_is_no_evidence_that_a_borderline_group_exists() -> None:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ann", find="Ann"),
        "address.city": ca("text", None, p_null=0.99),
    }
    script.clef = {"address.verified": noul(0.05)}  # the bool always resolves
    script.group_presence = {"address": 0.35}
    record = only(await run([Person], script, "Ann likes tea."))
    assert record.valid, record.errors
    assert record.value == Person(name="Ann")
    assert record.fields["address"].rejected_reason == "absent"


async def test_a_required_literal_is_no_evidence_either() -> None:
    class Mood(BaseModel):
        mood: Literal["happy", "sad"]
        note: Optional[str] = None

    class Holder(BaseModel):
        name: str
        mood: Optional[Mood] = None

    script = Script()
    script.clairic = {
        "name": ca("text", "Ann", find="Ann"),
        "mood.note": ca("text", None, p_null=0.99),
    }
    script.clef = {"mood.mood": pick("happy", ["happy", "sad"])}
    script.group_presence = {"mood": 0.3}
    record = only(await run([Holder], script, "Ann says hi."))
    assert record.valid and record.value == Holder(name="Ann")


async def test_the_union_fallback_is_decided_by_clairic_evidence() -> None:
    class Calm(BaseModel):
        name: Optional[str] = None
        quiet: bool = False
        still: bool = False

    class Loud(BaseModel):
        name: Optional[str] = None
        shouts: bool = False

    class Beast(BaseModel):
        kind: Union[Calm, Loud]

    script = Script()
    script.clairic = {
        "kind.Calm.name": ca("text", None, p_null=0.99),
        "kind.Loud.name": ca("text", "Rex", find="Rex"),
    }
    script.clef = {
        "kind": pick("Loud", ["Calm", "Loud"], 0.5),
        "kind.Calm.quiet": noul(0.05),  # False is the default: no evidence
        "kind.Calm.still": noul(0.05),
        "kind.Loud.shouts": noul(0.05),
    }
    record = only(await run([Beast], script, "Rex."))
    # Calm's bools say what they would say anyway, Loud has a name: the name wins
    assert isinstance(record.value, Beast) and isinstance(record.value.kind, Loud)


# --------------------------------------------------------------------------- #
# Review fixes: presence, scheduling, wording
# --------------------------------------------------------------------------- #


class Dropping:
    """Wraps a transport and leaves the answers of some Clef questions out."""

    def __init__(self, script: Script, kind: str) -> None:
        self.script = script
        self.kind = kind

    async def run(
        self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
    ) -> Any:
        import dataclasses

        response = await self.script.transport.run(model, payload, meta=meta)
        if meta.kind != self.kind or meta.field is None:
            return response
        body = dict(response.body)
        body["answers"] = {}
        return dataclasses.replace(response, body=body)


async def test_a_missing_presence_answer_lands_in_the_borderline_band() -> None:
    from trustcall_system_one import create_extractor

    script = book_script()
    # asked, not skipped; and with every leaf null it is judged absent
    script.clairic["author.address.city"] = ca("text", None, p_null=0.99)
    script.clairic["author.address.zip"] = ca("text", None, p_null=0.99)
    extractor = create_extractor([Book], transport=Dropping(script, "presence"))
    record = only(await extractor.ainvoke(TEXT))
    assert ["author.address.city", "author.address.zip"] in extract_fields(script)
    assert record.valid and book(record).author.address is None
    assert record.fields["author.address"].raw == {"noul": 0.35}


class Siblings(BaseModel):
    title: str
    home: Optional[Address] = None
    work: Optional[Address] = None
    mail: Optional[Address] = None


async def test_sibling_presence_questions_share_one_request() -> None:
    script = Script()
    script.clairic = {
        "title": ca("text", "Dune", find="Dune"),
        "home.city": ca("text", "Reno", find="Reno"),
        "home.zip": ca("text", None, p_null=0.99),
        "mail.city": ca("text", "Rome", find="Rome"),
        "mail.zip": ca("text", None, p_null=0.99),
    }
    script.group_presence = {"work": 0.0}
    result = await run([Siblings], script, "Dune: home Reno, mail Rome.")
    record = only(result)
    assert record.value == Siblings(
        title="Dune", home=Address(city="Reno"), mail=Address(city="Rome")
    )
    # the entity gate and one request for the three siblings
    calls = script.calls("presence")
    assert len(calls) == 2
    assert calls[1][2].field == "home,work,mail"
    assert sorted(calls[1][1]["questions"]) == [
        "home.__presence",
        "mail.__presence",
        "work.__presence",
    ]
    assert not any("work.city" in f for f in extract_fields(script))


async def test_the_chosen_member_is_asked_before_the_parents_clairic_pass_ends() -> (
    None
):
    import asyncio

    from trustcall_system_one.testing import FakeTransport, route

    script = owner_script()
    started = asyncio.Event()

    async def extract(model: str, payload: Any, meta: Any) -> Any:
        if "pet.Dog.tricks" in payload["fields"]:
            started.set()
        else:
            # the root's own Clairic pass cannot finish until the member is asked:
            # a member that waited for it would deadlock here
            await asyncio.wait_for(started.wait(), timeout=2)
        return script._extract(model, payload, meta)

    script.transport = FakeTransport(
        route(
            presence=script._presence,
            extract=extract,
            clef_fields=script._clef_fields,
            attribution=script._attribution,
        )
    )
    record = only(await run([Owner], script, OWNER_TEXT))
    assert record.valid and type(record.value.pet) is Dog  # type: ignore[union-attr]


async def test_container_entries_carry_the_containers_engine() -> None:
    script = book_script()
    script.group_presence = {"author.address": 0.05}
    record = only(await run([Book], script))
    assert record.fields["author.address"].engine == "submodel"
    assert record.fields["author.address"].raw == {"noul": 0.05}
    script = owner_script()
    record = only(await run([Owner], script, OWNER_TEXT))
    pet = record.fields["pet"]
    assert pet.engine == "union_models" and pet.raw["choice"] == "dog"
    leaves = [a for a in record.fields.values() if a.engine in ("clairic", "const")]
    assert {a.pretty for a in leaves} >= {"name", "pet.tricks", "pet.kind"}
    assert "pet" not in {a.pretty for a in leaves}


async def test_member_leaves_name_the_member_in_their_instructions() -> None:
    script = owner_script()
    await run([Owner], script, OWNER_TEXT)
    fields = {
        name: field
        for call in script.calls("extract")
        for name, field in call[1]["fields"].items()
    }
    assert fields["pet.Dog.tricks"]["instructions"] == (
        "Owner → pet (Dog): the Owner's pet's tricks"
    )
    assert fields["pet.Dog.nickname"]["instructions"] == (
        "Owner → pet (Dog): the Owner's pet's nickname"
    )


async def test_a_recursive_models_deferred_field_is_warned_about_once() -> None:
    class Tree(BaseModel):
        label: str
        parent: Optional["Tree"] = None
        children: list[Address] = []

    Tree.model_rebuild()
    script = Script()
    script.clairic = {"label": ca("text", "oak", find="oak")}
    script.group_presence = {"parent": 0.0}
    result = await run([Tree], script, "oak")
    mentions = [w for w in result.warnings if "collection fields" in w]
    assert len(mentions) == 1 and "Tree.children" in mentions[0]
    assert sum("max_recursion" in w for w in result.warnings) == 1


# --------------------------------------------------------------------------- #
# Clef answers as evidence, errors that go with a dropped group, alias dumps
# --------------------------------------------------------------------------- #


class Prefs(BaseModel):
    dark_mode: bool = False
    city: Optional[str] = None


class User(BaseModel):
    name: str
    prefs: Optional[Prefs] = None


async def test_a_clef_answer_that_differs_from_the_default_is_evidence() -> None:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ann", find="Ann"),
        "prefs.city": ca("text", None, p_null=0.99),
    }
    script.clef = {"prefs.dark_mode": noul(0.98)}
    script.group_presence = {"prefs": 0.4}
    record = only(await run([User], script, "Ann uses dark mode."))
    assert record.value == User(name="Ann", prefs=Prefs(dark_mode=True))
    assert record.fields["prefs"].rejected_reason is None


async def test_a_clef_answer_equal_to_the_default_is_not_evidence() -> None:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ann", find="Ann"),
        "prefs.city": ca("text", None, p_null=0.99),
    }
    script.clef = {"prefs.dark_mode": noul(0.02)}  # False, the default
    script.group_presence = {"prefs": 0.4}
    record = only(await run([User], script, "Ann likes tea."))
    assert record.value == User(name="Ann")
    assert record.fields["prefs"].rejected_reason == "absent"


class Mood(BaseModel):
    happy: Optional[bool]  # nullable, no default
    note: Optional[str] = None


class Feeling(BaseModel):
    name: str
    mood: Optional[Mood] = None


async def test_an_answered_nullable_clef_field_is_evidence_but_unstated_is_not() -> (
    None
):
    answered = Script()
    answered.clairic = {
        "name": ca("text", "Ann", find="Ann"),
        "mood.note": ca("text", None, p_null=0.99),
    }
    answered.clef = {"mood.happy": pick("yes", ["yes", "no", "unstated"])}
    answered.group_presence = {"mood": 0.4}
    record = only(await run([Feeling], answered, "Ann is happy."))
    assert record.value == Feeling(name="Ann", mood=Mood(happy=True))
    unstated = Script()
    unstated.clairic = dict(answered.clairic)
    unstated.clef = {"mood.happy": pick("unstated", ["yes", "no", "unstated"])}
    unstated.group_presence = {"mood": 0.4}
    record = only(await run([Feeling], unstated, "Ann is here."))
    assert record.value == Feeling(name="Ann")


async def test_a_clef_answer_differing_from_the_default_decides_a_union_fallback() -> (
    None
):
    class Calm(BaseModel):
        name: Optional[str] = None
        quiet: bool = False

    class Loud(BaseModel):
        name: Optional[str] = None
        shouts: bool = False

    class Beast(BaseModel):
        kind: Union[Calm, Loud]

    script = Script()
    script.clairic = {
        "kind.Calm.name": ca("text", None, p_null=0.99),
        "kind.Loud.name": ca("text", None, p_null=0.99),
    }
    script.clef = {
        "kind": pick("Loud", ["Calm", "Loud"], 0.5),
        "kind.Calm.quiet": noul(0.05),
        "kind.Loud.shouts": noul(0.98),  # True is not the default
    }
    record = only(await run([Beast], script, "It roars."))
    assert isinstance(record.value, Beast) and isinstance(record.value.kind, Loud)


async def test_a_required_clef_leaf_never_counts_even_when_it_resolves() -> None:
    class Strict(BaseModel):
        flag: bool  # required
        note: Optional[str] = None

    class Holder(BaseModel):
        name: str
        strict: Optional[Strict] = None

    script = Script()
    script.clairic = {
        "name": ca("text", "Ann", find="Ann"),
        "strict.note": ca("text", None, p_null=0.99),
    }
    script.clef = {"strict.flag": noul(0.99)}
    script.group_presence = {"strict": 0.4}
    record = only(await run([Holder], script, "Ann."))
    assert record.value == Holder(name="Ann")


class Cat2(BaseModel):
    kind: Literal["cat"]
    lives: int


class Dog2(BaseModel):
    kind: Literal["dog"]
    tricks: int


class Keeper2(BaseModel):
    name: Optional[str] = None
    pet: Union[Cat2, Dog2]


class Zoo2(BaseModel):
    title: str
    keeper: Optional[Keeper2] = None


async def test_errors_of_a_dropped_group_do_not_invalidate_the_record() -> None:
    script = Script()
    script.clairic = {
        "title": ca("text", "Zoo", find="Zoo"),
        "keeper.name": ca("text", "Sam", find="Sam"),
        "keeper.pet.Dog2.tricks": ca("number", None, p_null=0.99),
        "keeper.pet.Cat2.lives": ca("number", None, p_null=0.99),
    }
    script.clef = {"keeper.pet": pick("dog", ["cat", "dog"])}
    script.group_presence = {"keeper": 0.9}
    result = await run([Zoo2], script, "Zoo keeper Sam has a dog.")
    record = only(result)
    # the required union inside `keeper` cannot be built, so `keeper` is dropped:
    # that is all that is wrong, and the record stands
    assert record.valid, record.errors
    assert record.value == Zoo2(title="Zoo")
    assert record.errors == ()
    assert [
        r.path for r in record.rejected if r.reason == "invalid_optional_group"
    ] == ["/keeper"]


async def test_a_required_union_error_still_counts_in_a_kept_group() -> None:
    class Required(BaseModel):
        title: str
        keeper: Keeper2

    script = Script()
    script.clairic = {
        "title": ca("text", "Zoo", find="Zoo"),
        "keeper.name": ca("text", "Sam", find="Sam"),
        "keeper.pet.Dog2.tricks": ca("number", None, p_null=0.99),
        "keeper.pet.Cat2.lives": ca("number", None, p_null=0.99),
    }
    script.clef = {"keeper.pet": pick("dog", ["cat", "dog"])}
    record = only(await run([Required], script, "Zoo keeper Sam has a dog."))
    assert not record.valid
    assert record.errors == ("keeper.pet.tricks: Input should be a valid integer",)


class Inner2(BaseModel):
    code: str = Field(alias="Code")
    note: str = Field(alias="Note")


class Outer2(BaseModel):
    title: str = Field(alias="Title")
    inner: Optional[Inner2] = Field(default=None, alias="Inner")
    req: int = Field(alias="Req")


async def test_the_after_of_an_invalid_record_is_keyed_by_alias_throughout() -> None:
    script = Script()
    script.clairic = {
        "title": ca("text", "T", find="T"),
        "inner.code": ca("text", "X1", find="X1"),
        "inner.note": ca("text", "hi", find="hi"),
        "req": ca("number", None, p_null=0.99),
    }
    record = only(await run([Outer2], script, "T X1 hi"))
    assert not record.valid
    assert record.after == {
        "Title": "T",
        "Req": None,
        "Inner": {"Code": "X1", "Note": "hi"},
    }


async def test_a_typed_dict_member_in_after_stays_as_it_is() -> None:
    from typing_extensions import TypedDict

    class Geo(TypedDict):
        lat: float

    class Place(BaseModel):
        name: str
        geo: Optional[Geo] = None
        size: int

    script = Script()
    script.clairic = {
        "name": ca("text", "Oslo", find="Oslo"),
        "geo.lat": ca("number", 59.9),
        "size": ca("number", None, p_null=0.99),
    }
    record = only(await run([Place], script, "Oslo at 59.9."))
    assert not record.valid
    assert after(record)["geo"] == {"lat": 59.9}


class Unviable(BaseModel):
    code: str
    again: "Unviable"  # required and recursive: it can never be filled in


Unviable.model_rebuild()


class Holds(BaseModel):
    name: str
    extra: Optional[Unviable] = None


async def test_an_optional_group_that_can_never_validate_is_not_asked() -> None:
    from trustcall_system_one import compile_schema

    plan = compile_schema(Holds)
    spec = plan.spec(("extra",))
    assert spec.engine == "skip" and spec.warnings == ("unviable_group",)
    assert spec.clef is None and [g.pretty for g in plan.groups] == [""]
    assert sum("unviable_group" in w for w in plan.warnings) == 1
    script = Script()
    script.clairic = {"name": ca("text", "Ann", find="Ann")}
    result = await run([Holds], script, "Ann.")
    assert only(result).value == Holds(name="Ann")
    assert presence_fields(script) == [None]  # no question about `extra`
    assert extract_fields(script) == [["name"]]


# --------------------------------------------------------------------------- #
# Errors propagate as themselves from every concurrent step
# --------------------------------------------------------------------------- #

BOOM: list[Any] = [
    TransportError("boom"),
    ModelResponseError("bad response"),
    AssertionError("fake says no"),
]


@pytest.mark.parametrize("error", BOOM, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize(
    ("where", "schema"),
    [
        ("extract:title", Book),
        ("clef_fields:pet", Owner),
        ("extract:author.address.city", Book),  # a nested group
        ("extract:pet.Dog.tricks", Owner),  # a union member
        ("extract:name", Owner),
    ],
)
async def test_errors_propagate_as_themselves(
    error: Exception, where: str, schema: Any
) -> None:
    import asyncio

    from trustcall_system_one.testing import FakeTransport, route

    script = book_script() if schema is Book else owner_script()
    kind, name = where.split(":")

    def failing(handler: Any, mine: str) -> Any:
        async def run_handler(model: str, payload: Any, meta: Any) -> Any:
            fields = payload.get("fields") or payload.get("questions") or {}
            if mine == kind and name in fields:
                await asyncio.sleep(0)
                raise error
            await asyncio.sleep(0.02)  # the siblings are still pending
            return handler(model, payload, meta)

        return run_handler

    script.clef = {**script.clef}
    script.transport = FakeTransport(
        route(
            presence=failing(script._presence, "presence"),
            extract=failing(script._extract, "extract"),
            clef_fields=failing(script._clef_fields, "clef_fields"),
            attribution=failing(script._attribution, "attribution"),
        )
    )
    text = TEXT if schema is Book else OWNER_TEXT
    with pytest.raises(type(error)) as info:
        await make_extractor([schema], script).ainvoke(text)
    assert info.value is error or str(info.value) == str(error)
    if isinstance(error, TrustcallSystemOneError):
        assert cast(Any, info.value).calls  # the M3 context is attached


async def test_a_pending_clef_request_does_not_hide_a_transport_error() -> None:
    import asyncio

    from trustcall_system_one import create_extractor
    from trustcall_system_one.testing import FakeTransport, clef_response, route

    class Person(BaseModel):
        name: str
        active: bool

    def presence(model: str, payload: Any, meta: Any) -> Any:
        return clef_response({q: noul(0.95) for q in payload["questions"]})

    def extract(model: str, payload: Any, meta: Any) -> Any:
        raise TransportError("boom from extract")

    async def clef_fields(model: str, payload: Any, meta: Any) -> Any:
        await asyncio.sleep(0.05)
        return clef_response({q: noul(0.9) for q in payload["questions"]})

    transport = FakeTransport(
        route(presence=presence, extract=extract, clef_fields=clef_fields)
    )
    with pytest.raises(TransportError, match="boom from extract") as info:
        await create_extractor([Person], transport=transport).ainvoke("Ann is active.")
    assert cast(Any, info.value).calls and cast(Any, info.value).usage is not None


@pytest.mark.parametrize("schema", [Owner, Book], ids=lambda s: s.__name__)
async def test_an_incomplete_clairic_answer_is_not_a_cancelled_error(
    schema: Any,
) -> None:
    from trustcall_system_one.testing import FakeTransport, clairic_response, route

    script = book_script() if schema is Book else owner_script()

    def incomplete(model: str, payload: Any, meta: Any) -> Any:
        return clairic_response({})

    script.transport = FakeTransport(
        route(
            presence=script._presence,
            extract=incomplete,
            clef_fields=script._clef_fields,
            attribution=script._attribution,
        )
    )
    text = TEXT if schema is Book else OWNER_TEXT
    with pytest.raises(AssertionError, match="does not fit the request"):
        await make_extractor([schema], script).ainvoke(text)
