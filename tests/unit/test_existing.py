"""The ``existing`` argument: shapes, schema policy, key normalisation, cardinality.

The policy matrix is ported by behaviour from trustcall's
``test_strict_existing.py``; the tests are written for this library.
"""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import logging
import uuid
from typing import Any, Literal, Optional, Union

import pytest
from pydantic import AliasChoices, BaseModel, Field

from trustcall_system_one import SchemaInstance, compile_schema
from trustcall_system_one._existing import (
    ANY,
    ExistingSet,
    coerce_existing,
    fits,
    normalise_record,
    record_json,
    target_for,
)

KNOWN = ["DummySchema"]


class DummySchema(BaseModel):
    """A recognised schema."""

    foo: str
    bar: int


class Other(BaseModel):
    baz: str


def coerced(existing: Any, policy: Any = True, known: Any = KNOWN) -> ExistingSet:
    return coerce_existing(existing, known, policy)


def triples(es: ExistingSet) -> list[tuple[str, str, dict[str, Any]]]:
    return [(r.record_id, r.schema_name, r.data) for r in es.records]


# --------------------------------------------------------------------------- #
# The five shapes
# --------------------------------------------------------------------------- #

REC = {"foo": "val", "bar": 123}


def test_dict_form_uses_the_schema_name_as_the_id() -> None:
    es = coerced({"DummySchema": REC})
    assert triples(es) == [("DummySchema", "DummySchema", REC)]
    assert es.records[0].listed is False
    assert es.ids == ("DummySchema",)


def test_list_of_schema_instances() -> None:
    es = coerced([SchemaInstance("rid", "DummySchema", REC)])
    assert triples(es) == [("rid", "DummySchema", REC)]
    assert es.records[0].listed is True


def test_list_of_id_name_record_tuples() -> None:
    es = coerced([("rid7", "DummySchema", {"foo": "tuple", "bar": 200})])
    assert triples(es) == [("rid7", "DummySchema", {"foo": "tuple", "bar": 200})]


def test_list_of_id_model_pairs_names_the_schema_after_the_class() -> None:
    es = coerced([("rid", DummySchema(foo="a", bar=1))])
    assert triples(es) == [("rid", "DummySchema", {"foo": "a", "bar": 1})]


def test_bare_models_get_minted_ids() -> None:
    es = coerced([DummySchema(foo="a", bar=1), DummySchema(foo="b", bar=2)])
    first, second = es.records
    assert first.record_id != second.record_id
    assert uuid.UUID(first.record_id).version == 4
    assert (first.schema_name, first.data) == ("DummySchema", {"foo": "a", "bar": 1})


def test_a_model_record_is_dumped_by_alias_in_json_mode() -> None:
    class Dated(BaseModel):
        when: Any = Field(alias="whenAt")

    import datetime as dt

    es = coerced({"Dated": Dated(whenAt=dt.date(2025, 3, 4))}, known=["Dated"])
    assert es.records[0].data == {"whenAt": "2025-03-04"}


def test_a_dict_record_is_made_json_compatible() -> None:
    import datetime as dt
    from decimal import Decimal

    data = record_json(
        {"when": dt.date(2025, 3, 4), "price": Decimal("1.50")}, where="x"
    )
    assert data == {"when": "2025-03-04", "price": "1.50"}


def test_none_and_empty_containers_mean_no_records() -> None:
    empties: list[Any] = [None, {}, []]
    for empty in empties:
        es = coerced(empty)
        assert es.records == () and es.opaque == () and es.ids == ()


@pytest.mark.parametrize("policy", [True, False, "ignore"])
@pytest.mark.parametrize("bad", ["invalid_type", 42, object()])
def test_a_wrong_container_always_raises(policy: Any, bad: Any) -> None:
    with pytest.raises(ValueError, match="Invalid type for existing"):
        coerced(bad, policy)


@pytest.mark.parametrize("policy", [True, False, "ignore"])
@pytest.mark.parametrize(
    "item",
    [("only", "two", "strings"), ("rid", {"a": 1}), 7, "text", ("a",), ("rid", "S", 5)],
)
def test_a_wrong_item_always_raises(policy: Any, item: Any) -> None:
    with pytest.raises(ValueError):
        coerced([item], policy)


def test_record_ids_must_be_non_empty_strings_and_unique_per_schema() -> None:
    with pytest.raises(ValueError, match="record id"):
        coerced([(5, "DummySchema", REC)])
    with pytest.raises(ValueError, match="record id"):
        coerced([("", "DummySchema", REC)])
    with pytest.raises(ValueError, match="twice"):
        coerced([("a", "DummySchema", REC), ("a", "DummySchema", REC)])
    # the same id for two different schemas is fine
    coerced([("a", "DummySchema", REC), ("a", ANY, REC)])


def test_the_dict_form_holds_one_record_per_schema() -> None:
    with pytest.raises(ValueError, match="one record per schema"):
        coerced({"DummySchema": [REC, REC]})


# --------------------------------------------------------------------------- #
# existing_schema_policy: True / False / "ignore"
# --------------------------------------------------------------------------- #

UNKNOWN_DICT = {"UnknownSchema": {"foo": "val", "bar": 999}}
UNKNOWN_LIST = [SchemaInstance("rid2", "Unknown", {"some": "stuff"})]
UNKNOWN_TUPLE = [("rid8", "Unknown", {"data": "test"})]
UNKNOWN_PAIR = [("rid9", Other(baz="x"))]
UNKNOWN_BARE = [Other(baz="y")]


@pytest.mark.parametrize(
    "existing", [UNKNOWN_DICT, UNKNOWN_LIST, UNKNOWN_TUPLE, UNKNOWN_PAIR, UNKNOWN_BARE]
)
def test_policy_true_raises_for_an_unknown_schema(existing: Any) -> None:
    with pytest.raises(ValueError, match="doesn't match any schema"):
        coerced(existing, True)


def test_the_policy_error_names_the_known_schemas() -> None:
    with pytest.raises(ValueError, match="DummySchema"):
        coerced(UNKNOWN_DICT, True)


@pytest.mark.parametrize(
    "existing", [UNKNOWN_DICT, UNKNOWN_LIST, UNKNOWN_TUPLE, UNKNOWN_PAIR, UNKNOWN_BARE]
)
def test_policy_false_keeps_an_unknown_record_as_opaque_with_a_warning(
    existing: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        es = coerced(existing, False)
    assert es.records == ()
    (opaque,) = es.opaque
    assert es.ids == (opaque.record_id,)
    assert any("context only" in w and opaque.record_id in w for w in es.warnings)
    assert any("context only" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "existing", [UNKNOWN_DICT, UNKNOWN_LIST, UNKNOWN_TUPLE, UNKNOWN_PAIR, UNKNOWN_BARE]
)
def test_policy_ignore_drops_an_unknown_record_with_a_debug_log(
    existing: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        es = coerced(existing, "ignore")
    assert es == ExistingSet()
    assert any(
        r.levelno == logging.DEBUG and "ignoring" in r.message for r in caplog.records
    )
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("policy", [True, False, "ignore"])
@pytest.mark.parametrize(
    "existing",
    [
        {"DummySchema": REC},
        [SchemaInstance("rid", "DummySchema", REC)],
        [("rid", "DummySchema", REC)],
        [("rid", DummySchema(foo="val", bar=123))],
    ],
)
def test_a_known_schema_is_fine_under_every_policy(existing: Any, policy: Any) -> None:
    es = coerced(existing, policy)
    assert [r.schema_name for r in es.records] == ["DummySchema"]
    assert es.records[0].data == REC and es.opaque == ()


@pytest.mark.parametrize("policy", [True, False, "ignore"])
def test_any_is_known_under_every_policy(policy: Any) -> None:
    for existing in (
        {ANY: {"some": "stuff"}},
        [SchemaInstance("rid3", ANY, {"some": "stuff"})],
        [("rid3", ANY, {"some": "stuff"})],
    ):
        es = coerced(existing, policy)
        assert [r.schema_name for r in es.records] == [ANY]
        assert es.opaque == ()


@pytest.mark.parametrize(
    ("policy", "kept", "opaque"),
    [(False, ["rid4", "rid6"], ["rid5"]), ("ignore", ["rid4", "rid6"], [])],
)
def test_a_mixed_list_is_split_by_the_policy(
    policy: Any, kept: list[str], opaque: list[str]
) -> None:
    existing = [
        SchemaInstance("rid4", "DummySchema", REC),
        SchemaInstance("rid5", "Unknown", {"data": "test"}),
        SchemaInstance("rid6", ANY, {"other": "data"}),
    ]
    es = coerced(existing, policy)
    assert [r.record_id for r in es.records] == kept
    assert [r.record_id for r in es.opaque] == opaque
    with pytest.raises(ValueError):
        coerced(existing, True)


def test_the_mixed_dict_form_under_the_policies() -> None:
    existing = {
        "DummySchema": {"foo": "val2", "bar": 789},
        "RandomThing": {"some": "stuff2"},
        ANY: {"pass_through": True},
    }
    es = coerced(existing, False)
    assert [r.schema_name for r in es.records] == ["DummySchema", ANY]
    assert [r.schema_name for r in es.opaque] == ["RandomThing"]
    assert [r.schema_name for r in coerced(existing, "ignore").records] == [
        "DummySchema",
        ANY,
    ]


# --------------------------------------------------------------------------- #
# Which records a schema updates
# --------------------------------------------------------------------------- #


def test_a_schema_takes_its_own_records_else_the_any_ones() -> None:
    es = coerced([("a", ANY, REC), ("b", "DummySchema", REC)])
    assert [r.record_id for r in es.for_schema("DummySchema")] == ["b"]
    assert [r.record_id for r in es.for_schema("Other")] == ["a"]
    assert (
        coerced({"Other": {"baz": "x"}}, known=["Other", "DummySchema"]).for_schema(
            "DummySchema"
        )
        == []
    )


def test_dict_form_is_the_target_of_a_one_schema() -> None:
    es = coerced({"DummySchema": REC})
    target = target_for("DummySchema", es)
    assert target is not None and target.record_id == "DummySchema"


def test_an_explicit_one_takes_the_single_list_form_record() -> None:
    es = coerced([("rid", "DummySchema", REC)])
    target = target_for("DummySchema", es)
    assert target is not None and target.record_id == "rid"


def test_one_with_several_records_suggests_many() -> None:
    es = coerced([("a", "DummySchema", REC), ("b", "DummySchema", REC)])
    with pytest.raises(ValueError, match="Many"):
        target_for("DummySchema", es)


def test_no_record_means_no_target() -> None:
    assert target_for("DummySchema", ExistingSet()) is None
    es = coerced({"Other": {"baz": "x"}}, known=["Other", "DummySchema"])
    assert target_for("DummySchema", es) is None


# --------------------------------------------------------------------------- #
# Key normalisation
# --------------------------------------------------------------------------- #


class Address(BaseModel):
    city: str
    postal_code: Optional[str] = Field(default=None, alias="postalCode")


class Cat(BaseModel):
    kind: Literal["cat"] = "cat"
    nick_name: str = Field(alias="nickName")


class Dog(BaseModel):
    kind: Literal["dog"] = "dog"
    dog_name: str = Field(alias="dogName")


class Person(BaseModel):
    full_name: str = Field(alias="fullName")
    nick: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("nickname", "nick")
    )
    home: Optional[Address] = None
    pet: Optional[Union[Cat, Dog]] = None
    extra: Optional[str] = None


def plan() -> Any:
    return compile_schema(Person)


def test_field_names_become_validation_keys_at_every_level() -> None:
    record = normalise_record(
        plan(),
        {
            "full_name": "Ada",
            "nick": "A",
            "home": {"city": "Rome", "postal_code": "00100"},
            "extra": "x",
        },
    )
    assert record == {
        "fullName": "Ada",
        "nickname": "A",
        "home": {"city": "Rome", "postalCode": "00100"},
        "extra": "x",
    }


def test_keys_already_in_validation_form_stay() -> None:
    data = {"fullName": "Ada", "home": {"city": "Rome", "postalCode": "00100"}}
    assert normalise_record(plan(), data) == data


def test_unknown_keys_are_kept_and_the_validation_key_wins_a_clash() -> None:
    record = normalise_record(
        plan(), {"full_name": "old", "fullName": "new", "mystery": 1}
    )
    assert record == {"fullName": "new", "mystery": 1}


def test_a_union_value_is_normalised_as_the_member_it_validates_as() -> None:
    cat = normalise_record(plan(), {"full_name": "A", "pet": {"nick_name": "Tom"}})
    assert cat["pet"] == {"nickName": "Tom"}
    dog = normalise_record(
        plan(), {"full_name": "A", "pet": {"kind": "dog", "dog_name": "Rex"}}
    )
    assert dog["pet"] == {"kind": "dog", "dogName": "Rex"}


def test_a_value_that_is_not_an_object_is_left_alone() -> None:
    record = normalise_record(plan(), {"full_name": "A", "home": None})
    assert record == {"fullName": "A", "home": None}


# --------------------------------------------------------------------------- #
# "__any__" claims and untouched records
# --------------------------------------------------------------------------- #


def test_a_schema_sees_an_any_record_only_when_it_claimed_it() -> None:
    es = coerced(
        [("a", ANY, REC), ("b", "DummySchema", REC)], known=["DummySchema", "Other"]
    )
    claimed = es.without_unclaimed_any("Other", {"a": "Other"})
    assert [r.record_id for r in claimed.for_schema("Other")] == ["a"]
    unclaimed = es.without_unclaimed_any("Other", {"a": "DummySchema"})
    assert unclaimed.for_schema("Other") == []
    # a schema's own records are always seen
    assert [r.record_id for r in unclaimed.named("DummySchema")] == ["b"]
    assert unclaimed.opaque == es.opaque


def test_untouched_is_keyed_by_schema_and_id() -> None:
    es = coerced(
        [("same", "DummySchema", REC), ("same", ANY, REC), ("x", "Unknown", REC)],
        False,
    )
    assert es.untouched(set()) == ("same", "same", "x")
    assert es.untouched({("DummySchema", "same")}) == ("same", "x")
    assert es.untouched({(ANY, "same"), ("Unknown", "x")}) == ("same",)


def test_fits_normalises_the_record_before_validating() -> None:
    class Aliased(BaseModel):
        full_name: str = Field(alias="fullName")

    plan = compile_schema(Aliased)
    es = coerced([("a", ANY, {"full_name": "Ada"}), ("b", ANY, {"nope": 1})])
    first, second = es.records
    assert fits(plan, first) and not fits(plan, second)


def test_the_items_of_collections_and_mappings_are_normalised_too() -> None:
    class Part(BaseModel):
        part_no: str = Field(alias="partNo")
        count: int = 1

    class Kit(BaseModel):
        parts: list[Part] = []
        spares: dict[str, Part] = {}
        stock: dict[str, int] = {}

    record = normalise_record(
        compile_schema(Kit),
        {
            "parts": [{"part_no": "A", "count": 2}, "junk"],
            "spares": {"x": {"part_no": "B"}},
            "stock": {"a": 1},
        },
    )
    assert record == {
        "parts": [{"partNo": "A", "count": 2}, "junk"],
        "spares": {"x": {"partNo": "B"}},
        "stock": {"a": 1},
    }
