"""Tests for the schema compiler: one case per row of the type table."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import datetime as dt
import enum
from decimal import Decimal
from typing import Annotated, Any, Literal, NewType, Optional, Union

import pytest
import typing_extensions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    create_model,
)
from typing_extensions import NotRequired, TypeAliasType, TypedDict

from trustcall_system_one import (
    FieldSpec,
    GateConfig,
    Limits,
    SchemaPlan,
    compile_schema,
    s1_field,
)
from trustcall_system_one._compile import render_instructions
from trustcall_system_one._types import UNSTATED
from trustcall_system_one._wire import ChoiceQuestion, ClairicRequest, ClefRequest

DESC = "Thing"
REQUIRED_THRESHOLD = 0.7
OPTIONAL_THRESHOLD = 0.3


def spec_for(annotation: Any, default: Any = ..., **field_kwargs: Any) -> FieldSpec:
    """Compile a one-field model and return the field's spec."""
    field_kwargs.setdefault("description", DESC)
    model = create_model("M", f=(annotation, Field(default, **field_kwargs)))
    return compile_schema(model).specs[0]


def clairic_dict(spec: FieldSpec) -> dict[str, Any]:
    assert spec.engine == "clairic"
    assert spec.clairic is not None
    return spec.clairic.model_dump(exclude_none=True)


def clef_dict(spec: FieldSpec) -> dict[str, Any]:
    assert spec.engine == "clef"
    assert spec.clef is not None
    return spec.clef.model_dump(exclude_none=True)


def text(instructions: str = DESC, **extra: Any) -> dict[str, Any]:
    return {
        "type": "text",
        "instructions": instructions,
        "null_threshold": REQUIRED_THRESHOLD,
        **extra,
    }


def numeric(kind: str = "number", **extra: Any) -> dict[str, Any]:
    spec = {
        "type": kind,
        "instructions": extra.pop("instructions", DESC),
        "null_threshold": REQUIRED_THRESHOLD,
        **extra,
    }
    if kind == "list":  # the API refuses a null_threshold on a list field
        del spec["null_threshold"]
    return spec


def make_enum(name: str, size: int) -> Any:
    factory: Any = enum.Enum
    return factory(name, {f"m{i}": i for i in range(size)})


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Size(str, enum.Enum):
    SMALL = "s"
    LARGE = "l"


class Rank(enum.IntEnum):
    LOW = 1
    HIGH = 2


class Impact(enum.Enum):
    """Impact levels, lowest first."""

    LOW = ("low", "Barely noticeable")
    MEDIUM = ("medium", "Noticeable but tolerable")
    HIGH = ("high", "Severe")

    def __new__(cls, value: str, description: str) -> "Impact":
        member = object.__new__(cls)
        member._value_ = value
        member.description = description  # type: ignore[attr-defined]
        return member


class Pet(BaseModel):
    name: str


class Cat(BaseModel):
    kind: Literal["cat"]
    lives: int


class Dog(BaseModel):
    kind: Literal["dog"]
    tricks: int


# --------------------------------------------------------------------------- #
# Clairic rows
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("annotation", "field_kwargs", "expected"),
    [
        pytest.param(str, {}, text(), id="str"),
        pytest.param(
            Annotated[str, Field(min_length=2, max_length=5, pattern="^a")],
            {},
            text("Thing. Between 2 and 5 characters. Must match the pattern ^a."),
            id="str-constraints",
        ),
        pytest.param(int, {}, numeric(integer=True), id="int"),
        pytest.param(float, {}, numeric(), id="float"),
        pytest.param(
            float,
            {"json_schema_extra": s1_field(decimals=3)},
            numeric(decimals=3),
            id="float-decimals",
        ),
        pytest.param(
            Annotated[float, Field(decimal_places=2)],
            {},
            numeric(decimals=2),
            id="float-decimal-places",
        ),
        pytest.param(Decimal, {}, numeric(), id="decimal"),
        pytest.param(
            Decimal,
            {"json_schema_extra": s1_field(decimals=5)},
            numeric(decimals=5),
            id="decimal-option",
        ),
        pytest.param(
            Annotated[Decimal, Field(decimal_places=4)],
            {},
            numeric(decimals=4),
            id="decimal-places",
        ),
        pytest.param(dt.date, {}, numeric("date"), id="date"),
        pytest.param(dt.time, {}, numeric("time"), id="time"),
        pytest.param(dt.datetime, {}, numeric("datetime"), id="datetime"),
        pytest.param(dt.timedelta, {}, numeric("duration", unit="seconds"), id="td"),
        pytest.param(
            dt.timedelta,
            {"json_schema_extra": s1_field(unit="hours")},
            numeric("duration", unit="hours"),
            id="td-unit",
        ),
        pytest.param(list[str], {}, numeric("list", items="text"), id="list-str"),
        pytest.param(list[int], {}, numeric("list", items="number"), id="list-int"),
        pytest.param(list[float], {}, numeric("list", items="number"), id="list-float"),
        pytest.param(list[dt.date], {}, numeric("list", items="date"), id="list-date"),
        pytest.param(set[int], {}, numeric("list", items="number"), id="set-int"),
        pytest.param(
            frozenset[str], {}, numeric("list", items="text"), id="frozenset-str"
        ),
        pytest.param(
            list[dt.datetime], {}, numeric("list", items="text"), id="list-datetime"
        ),
        pytest.param(list[dt.time], {}, numeric("list", items="text"), id="list-time"),
        pytest.param(
            list[dt.timedelta], {}, numeric("list", items="text"), id="list-timedelta"
        ),
        pytest.param(
            list[Literal["a", "b"]],
            {},
            numeric(
                "list",
                items="text",
                instructions="Thing. Each item must be one of: a, b.",
            ),
            id="list-literal",
        ),
        pytest.param(
            list[Color],
            {},
            numeric(
                "list",
                items="text",
                instructions="Thing. Each item must be one of: red, blue.",
            ),
            id="list-enum",
        ),
        pytest.param(
            str,
            {"json_schema_extra": s1_field(clairic_type="number")},
            numeric("number"),
            id="clairic-type-override",
        ),
        pytest.param(
            int,
            {"json_schema_extra": s1_field(integer=False, unit="USD")},
            numeric(integer=False, unit="USD"),
            id="number-options",
        ),
    ],
)
def test_clairic_rows(
    annotation: Any, field_kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    spec = spec_for(annotation, **field_kwargs)
    assert clairic_dict(spec) == expected
    assert spec.clef is None
    assert spec.wire_name == "f"
    assert spec.instructions == expected["instructions"]


def test_a_list_field_never_carries_a_null_threshold() -> None:
    """The live API answers HTTP 422 to ``list`` + ``null_threshold`` (error 3030)."""
    for annotation in (list[str], list[int], tuple[str, ...], set[int], list[Color]):
        spec = spec_for(annotation)
        assert "null_threshold" not in clairic_dict(spec)
        assert spec.clairic is not None and spec.clairic.null_threshold is None
        # the spec itself still holds the threshold the answer is judged by
        assert spec.null_threshold == REQUIRED_THRESHOLD
    # an explicit s1_field threshold does not get onto the wire either
    spec = spec_for(list[str], json_schema_extra=s1_field(null_threshold=0.9))
    assert "null_threshold" not in clairic_dict(spec)
    # every other Clairic type still has one
    for plain in (str, int, dt.date, float):
        assert clairic_dict(spec_for(plain))["null_threshold"] == (REQUIRED_THRESHOLD)


def test_decimal_places_over_six_are_clamped_with_a_warning() -> None:
    spec = spec_for(Annotated[Decimal, Field(decimal_places=9)])
    assert clairic_dict(spec)["decimals"] == 6
    assert "decimals_clamped" in spec.warnings


@pytest.mark.parametrize(
    ("annotation", "kwargs", "post", "item_kind"),
    [
        (str, {}, "", None),
        (int, {}, "int", None),
        (float, {}, "float", None),
        (Decimal, {}, "decimal", None),
        (dt.date, {}, "", None),
        (dt.timedelta, {}, "timedelta:seconds", None),
        (
            dt.timedelta,
            {"json_schema_extra": s1_field(unit="hours")},
            "timedelta:hours",
            None,
        ),
        (list[str], {}, "", "str"),
        (list[int], {}, "", "int"),
        (list[float], {}, "", "float"),
        (list[Decimal], {}, "", "decimal"),
        (list[dt.date], {}, "", "date"),
        (set[int], {}, "set", "int"),
        (frozenset[str], {}, "frozenset", "str"),
        (list[dt.datetime], {}, "list[datetime]", "datetime"),
        (list[dt.time], {}, "list[time]", "time"),
        (list[dt.timedelta], {}, "list[timedelta]", "timedelta:seconds"),
        (set[dt.datetime], {}, "set", "datetime"),
        (list[Color], {}, "", "choice"),
    ],
)
def test_post_tags_and_item_kinds(
    annotation: Any, kwargs: dict[str, Any], post: str, item_kind: str | None
) -> None:
    spec = spec_for(annotation, **kwargs)
    assert spec.post == post
    assert spec.item_kind == item_kind


@pytest.mark.parametrize(
    "annotation", [list[dt.datetime], list[dt.time], list[dt.timedelta]]
)
def test_lossy_lists_carry_a_warning(annotation: Any) -> None:
    spec = spec_for(annotation)
    assert spec.warnings == ("lossy_list",)
    assert spec_for(list[str]).warnings == ()


def test_months_and_years_are_marked_lossy() -> None:
    for unit in ("months", "years"):
        spec = spec_for(dt.timedelta, json_schema_extra=s1_field(unit=unit))
        assert "lossy_duration" in spec.warnings
    assert spec_for(dt.timedelta).warnings == ()


def test_list_of_literals_and_enums_records_element_choices() -> None:
    spec = spec_for(list[Literal["in progress", "done"]])
    assert dict(spec.element_choices or {}) == {
        "in_progress": "in progress",
        "done": "done",
    }
    spec = spec_for(set[Color])
    assert dict(spec.element_choices or {}) == {"RED": "red", "BLUE": "blue"}
    assert spec.post == "set"
    assert spec.clef is None


def test_estimate_for_bounded_numbers_is_opt_in() -> None:
    annotation = Annotated[float, Field(ge=0, le=10)]
    model = create_model("M", f=(annotation, Field(..., description=DESC)))
    plain = compile_schema(model).specs[0]
    assert clairic_dict(plain)["type"] == "number"
    assert "min" not in clairic_dict(plain)
    assert "max" not in clairic_dict(plain)
    plan = compile_schema(model, limits=Limits(estimate_for_bounded_numbers=True))
    assert clairic_dict(plan.specs[0]) == {
        "type": "estimate",
        "instructions": "Thing. Must be at least 0 and at most 10.",
        "min": 0.0,
        "max": 10.0,
        "null_threshold": REQUIRED_THRESHOLD,
    }


def test_one_sided_bounds_never_become_estimates() -> None:
    model = create_model(
        "M", f=(Annotated[int, Field(ge=0)], Field(..., description=DESC))
    )
    plan = compile_schema(model, limits=Limits(estimate_for_bounded_numbers=True))
    assert clairic_dict(plan.specs[0])["type"] == "number"


def test_integer_estimate_uses_integer_bounds() -> None:
    model = create_model(
        "M", f=(Annotated[int, Field(gt=0, lt=9)], Field(..., description=DESC))
    )
    plan = compile_schema(model, limits=Limits(estimate_for_bounded_numbers=True))
    payload = clairic_dict(plan.specs[0])
    assert (payload["type"], payload["min"], payload["max"]) == ("estimate", 0, 9)
    assert plan.specs[0].post == "int"


def test_invalid_clairic_type_override_names_the_field() -> None:
    with pytest.raises(ValueError, match="'f'"):
        spec_for(str, json_schema_extra=s1_field(clairic_type="estimate"))


# --------------------------------------------------------------------------- #
# Clef rows
# --------------------------------------------------------------------------- #


def test_bool_is_a_noul_question() -> None:
    spec = spec_for(bool)
    assert spec.engine == "clef"
    assert spec.clairic is None
    assert spec.wire_name == spec.clef_name == "f"
    assert clef_dict(spec) == {
        "type": "noul",
        "instructions": "Does the input say this is true: Thing?",
        "criteria": {
            "true": 'The input states or clearly implies: "Thing".',
            "false": 'The input states or clearly implies the opposite of: "Thing".',
        },
    }
    assert spec.key_map is None


def test_noul_criteria_can_be_overridden() -> None:
    spec = spec_for(
        bool, json_schema_extra=s1_field(criteria={"true": "yes!", "false": "no!"})
    )
    assert clef_dict(spec)["criteria"] == {"true": "yes!", "false": "no!"}


@pytest.mark.parametrize(
    "annotation",
    [Optional[bool], Union[bool, None], bool | None],
)
def test_optional_bool_is_a_three_way_choice(annotation: Any) -> None:
    spec = spec_for(annotation)
    assert clef_dict(spec) == {
        "type": "choice",
        "instructions": "Pick the option that best fits: Thing",
        "criteria": {
            "yes": "The input says this is true.",
            "no": "The input says this is false.",
            "unstated": "The input does not state this.",
        },
    }
    assert dict(spec.key_map or {}) == {
        "yes": True,
        "no": False,
        "unstated": UNSTATED,
    }
    assert spec.required is False
    assert spec.nullable is True


def test_optional_bool_can_stay_noul() -> None:
    spec = spec_for(Optional[bool], json_schema_extra=s1_field(clef="noul"))
    assert clef_dict(spec)["type"] == "noul"


def test_bool_can_be_forced_to_a_choice() -> None:
    spec = spec_for(bool, json_schema_extra=s1_field(clef="choice"))
    assert clef_dict(spec)["type"] == "choice"


def test_literal_is_a_choice_with_sanitised_option_ids() -> None:
    spec = spec_for(Literal["a", "b c", "d/e"])
    assert clef_dict(spec) == {
        "type": "choice",
        "instructions": "Pick the option that best fits: Thing",
        "criteria": {"a": "a", "b_c": "b c", "d_e": "d/e"},
    }
    assert dict(spec.key_map or {}) == {"a": "a", "b_c": "b c", "d_e": "d/e"}


def test_option_id_collisions_get_a_suffix() -> None:
    spec = spec_for(Literal["a b", "a_b", "a/b"])
    assert list(spec.key_map or {}) == ["a_b", "a_b_2", "a_b_3"]
    assert dict(spec.key_map or {}) == {
        "a_b": "a b",
        "a_b_2": "a_b",
        "a_b_3": "a/b",
    }


def test_literal_with_non_string_values_maps_back_to_the_values() -> None:
    spec = spec_for(Literal[1, 2, True])
    assert dict(spec.key_map or {}) == {"1": 1, "2": 2, "True": True}


def test_plain_enum_is_a_choice_mapping_back_to_values() -> None:
    spec = spec_for(Color)
    assert clef_dict(spec)["criteria"] == {"RED": "RED (red)", "BLUE": "BLUE (blue)"}
    assert dict(spec.key_map or {}) == {"RED": "red", "BLUE": "blue"}


def test_str_enum_and_int_enum_are_choices() -> None:
    assert dict(spec_for(Size).key_map or {}) == {"SMALL": "s", "LARGE": "l"}
    assert dict(spec_for(Rank).key_map or {}) == {"LOW": 1, "HIGH": 2}
    assert spec_for(Rank).engine == "clef"


def test_enum_member_descriptions_become_criteria() -> None:
    spec = spec_for(Impact)
    assert clef_dict(spec)["criteria"] == {
        "LOW": "LOW (low): Barely noticeable",
        "MEDIUM": "MEDIUM (medium): Noticeable but tolerable",
        "HIGH": "HIGH (high): Severe",
    }


def test_member_docstring_is_used_when_assigned() -> None:
    class Mood(enum.Enum):
        GOOD = "good"
        BAD = "bad"

    Mood.GOOD.__doc__ = "Things went well"
    spec = spec_for(Mood)
    assert clef_dict(spec)["criteria"] == {
        "GOOD": "GOOD (good): Things went well",
        "BAD": "BAD (bad)",
    }


def test_choice_criteria_can_be_overridden_by_id_or_value() -> None:
    spec = spec_for(
        Literal["a", "b c"],
        json_schema_extra=s1_field(criteria={"a": "first", "b c": "second"}),
    )
    assert clef_dict(spec)["criteria"] == {"a": "first", "b_c": "second"}


def test_ordered_enum_can_be_a_score() -> None:
    spec = spec_for(Impact, json_schema_extra=s1_field(clef="score"))
    assert clef_dict(spec) == {
        "type": "score",
        "instructions": "Rate on the scale, lowest level first: Thing",
        "criteria": ["Barely noticeable", "Noticeable but tolerable", "Severe"],
    }
    assert dict(spec.key_map or {}) == {"0": "low", "1": "medium", "2": "high"}


def test_score_levels_fall_back_to_values_and_accept_overrides() -> None:
    spec = spec_for(
        Literal["bad", "ok", "good"], json_schema_extra=s1_field(clef="score")
    )
    assert clef_dict(spec)["criteria"] == ["bad", "ok", "good"]
    spec = spec_for(
        Rank, json_schema_extra=s1_field(clef="score", criteria=["worst", "best"])
    )
    assert clef_dict(spec)["criteria"] == ["worst", "best"]
    with pytest.raises(ValueError, match="levels"):
        spec_for(Rank, json_schema_extra=s1_field(clef="score", criteria=["only"]))


@pytest.mark.parametrize("levels", [1, 11])
def test_score_needs_two_to_ten_levels(levels: int) -> None:
    members = make_enum("Levels", levels)
    with pytest.raises(ValueError, match=r"'f'.*2 to 10"):
        spec_for(members, json_schema_extra=s1_field(clef="score"))


def test_more_than_255_options_is_an_error_naming_the_field() -> None:
    big = make_enum("Big", 256)
    with pytest.raises(ValueError, match=r"'f'.*256 options"):
        spec_for(big)
    ok = make_enum("Ok", 255)
    assert len(spec_for(ok).key_map or {}) == 255


def test_single_option_literal_on_a_required_field_is_a_constant() -> None:
    spec = spec_for(Literal["cat"])
    assert spec.engine == "const"
    assert spec.clairic is None and spec.clef is None
    assert spec.wire_name == "" and spec.warnings == ()
    assert dict(spec.key_map or {}) == {"cat": "cat"}
    fields: dict[str, Any] = {"kind": (Literal["cat"], ...), "n": (int, ...)}
    plan = compile_schema(create_model("M", **fields))
    assert list(plan.clairic_fields()) == ["n"]
    assert plan.clef_questions() == {}


def test_single_option_with_a_default_or_nullable_is_skipped_with_a_warning() -> None:
    for default in ("cat", None):
        spec = spec_for(Optional[Literal["cat"]], default)
        assert spec.engine == "skip"
        assert spec.warnings == ("single_option",)
    spec = spec_for(Literal["cat"], "cat")
    assert (spec.engine, spec.warnings) == ("skip", ("single_option",))


def test_single_member_enum_on_a_required_field_is_a_constant() -> None:
    class One(enum.Enum):
        ONLY = 7

    spec = spec_for(One)
    assert spec.engine == "const"
    assert dict(spec.key_map or {}) == {"ONLY": 7}


def test_clef_override_is_rejected_where_it_cannot_apply() -> None:
    with pytest.raises(ValueError, match="only applies"):
        spec_for(str, json_schema_extra=s1_field(clef="choice"))
    with pytest.raises(ValueError, match="cannot"):
        spec_for(Color, json_schema_extra=s1_field(clef="noul"))
    with pytest.raises(ValueError, match="scored"):
        spec_for(bool, json_schema_extra=s1_field(clef="score"))


def test_union_of_primitives_is_text_plus_a_type_choice() -> None:
    spec = spec_for(Union[int, str])
    assert spec.engine == "clairic"
    assert clairic_dict(spec) == text()
    assert spec.wire_name == "f"
    assert spec.clef_name == "f.__type"
    assert spec.clef is not None
    assert spec.clef.model_dump(exclude_none=True) == {
        "type": "choice",
        "instructions": "What kind of value is this: Thing?",
        "criteria": {"int": "A whole number.", "str": "Plain text."},
    }
    assert dict(spec.key_map or {}) == {"int": "int", "str": "str"}
    assert spec.post == "union"


def test_pep604_union_and_plan_keys() -> None:
    model = create_model("M", f=(int | str, Field(..., description=DESC)))
    plan = compile_schema(model)
    assert list(plan.clairic_fields()) == ["f"]
    assert list(plan.clef_questions()) == ["f.__type"]
    assert plan.clef_names.path_to_name == {("f", "__type"): "f.__type"}
    assert plan.clairic_names.path_to_name == {("f",): "f"}


def test_union_of_primitives_dedupes_member_names() -> None:
    spec = spec_for(Union[int, float, str])
    assert list(spec.key_map or {}) == ["int", "float", "str"]


def test_union_with_unsupported_members_is_skipped_with_a_warning() -> None:
    spec = spec_for(Union[int, list[str]])
    assert spec.engine == "skip"
    assert spec.warnings == ("unsupported_union",)
    assert spec_for(Union[Pet, int]).warnings == ("unsupported_union",)


def test_union_of_literals_merges_into_one_choice() -> None:
    spec = spec_for(Union[Literal["a"], Literal["b"]])
    assert list(spec.key_map or {}) == ["a", "b"]


# --------------------------------------------------------------------------- #
# Skipped and deferred kinds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "annotation",
    [Any, object, dict, dict[str, Any], Optional[dict[str, Any]]],
)
def test_any_like_fields_are_skipped(annotation: Any) -> None:
    spec = spec_for(annotation)
    assert spec.engine == "skip"
    assert spec.clairic is None
    assert spec.clef is None
    assert spec.wire_name == ""


@pytest.mark.parametrize("annotation", [Any, object, dict, dict[str, Any]])
def test_any_policy_text_asks_for_text(annotation: Any) -> None:
    model = create_model("M", f=(annotation, Field(..., description=DESC)))
    plan = compile_schema(model, limits=Limits(any_policy="text"))
    assert clairic_dict(plan.specs[0]) == text()


def test_skip_option_skips_a_field() -> None:
    spec = spec_for(str, "x", json_schema_extra=s1_field(skip=True))
    assert spec.engine == "skip"
    assert spec.clairic is None


@pytest.mark.parametrize(
    ("annotation", "engine"),
    [
        (list[Pet], "collection"),
        (set[Pet], "collection"),
        (dict[str, int], "mapping"),
        (dict[str, str], "mapping"),
    ],
)
def test_deferred_kinds_are_recorded_not_expanded(annotation: Any, engine: str) -> None:
    spec = spec_for(annotation)
    assert spec.engine == engine
    assert spec.clairic is None
    assert spec.clef is None
    assert spec.warnings == ()


def test_deferred_lists_the_deferred_specs() -> None:
    class Owner(BaseModel):
        name: str
        pet: Pet
        pets: list[Pet]
        scores: dict[str, int]
        any_pet: Cat | Dog
        age: int

    plan = compile_schema(Owner)
    # sub-models and unions of models are expanded; collections and mappings wait
    assert [spec.pretty for spec in plan.collection_specs()] == ["pets", "scores"]
    assert list(plan.clairic_fields()) == [
        "name",
        "pet.name",
        "age",
        "any_pet.Cat.lives",
        "any_pet.Dog.tricks",
    ]


def test_models_are_resolved_and_stored_on_the_spec() -> None:
    class Owner(BaseModel):
        pet: Pet
        pets: list[Pet]
        any_pet: Cat | Dog
        many: list[Cat | Dog]
        name: str

    plan = compile_schema(Owner)
    assert plan.spec(("pet",)).models == (Pet,)
    assert plan.spec(("pets",)).models == (Pet,)
    assert plan.spec(("any_pet",)).models == (Cat, Dog)
    assert plan.spec(("many",)).models == (Cat, Dog)
    assert plan.spec(("name",)).models == ()


def test_recursive_models_store_themselves() -> None:
    class Node(BaseModel):
        children: list["Node"] = []
        parent: Optional["Node"] = None

    Node.model_rebuild()
    plan = compile_schema(Node)
    assert plan.spec(("children",)).models == (Node,)
    assert plan.spec(("parent",)).models == (Node,)


def test_function_local_models_resolve_through_the_defining_namespace() -> None:
    class Pet(BaseModel):  # shadows the module-level Pet
        owner_tag: int

    class Owner(BaseModel):
        pets: list["Pet"]
        pet: "Pet"

    Owner.model_rebuild()
    plan = compile_schema(Owner)
    assert plan.spec(("pets",)).models == (Pet,)
    assert plan.spec(("pets",)).models[0] is not globals()["Pet"]
    assert plan.spec(("pet",)).models == (Pet,)


def _with_string_annotation(model: type[BaseModel], name: str, annotation: Any) -> None:
    """Leave a reference unresolved, as pydantic does on Python 3.10."""
    model.model_fields[name].annotation = annotation


def test_unresolved_string_references_use_the_defining_namespace_first() -> None:
    class Pet(BaseModel):  # shadows the module-level Pet
        owner_tag: int

    class Cat(BaseModel):  # shadows the module-level Cat
        whiskers: int

    class Owner(BaseModel):
        pets: list[int]
        many: list[int]

    _with_string_annotation(Owner, "pets", list["Pet"])
    _with_string_annotation(Owner, "many", list[Union["Cat", "Dog"]])
    plan = compile_schema(Owner)
    assert plan.spec(("pets",)).engine == "collection"
    assert plan.spec(("pets",)).models == (Pet,)
    assert plan.spec(("pets",)).models[0] is not globals()["Pet"]
    assert plan.spec(("many",)).models == (Cat, globals()["Dog"])  # Dog: module


def test_unresolved_string_references_fall_back_to_the_module() -> None:
    class Owner(BaseModel):
        pets: list[int]
        any_pet: int

    _with_string_annotation(Owner, "pets", list["Pet"])
    _with_string_annotation(Owner, "any_pet", Union["Cat", "Dog"])
    plan = compile_schema(Owner)
    assert plan.spec(("pets",)).models == (globals()["Pet"],)
    assert plan.spec(("any_pet",)).engine == "union_models"
    assert plan.spec(("any_pet",)).models == (Cat, Dog)


def test_unresolvable_string_references_are_skipped() -> None:
    class Owner(BaseModel):
        pets: list[int]

    _with_string_annotation(Owner, "pets", eval("list['Nowhere']"))
    assert compile_schema(Owner).spec(("pets",)).engine == "skip"


def test_unsupported_types_are_skipped_with_a_warning() -> None:
    assert spec_for(bytes).warnings == ("unsupported_type",)
    assert spec_for(dict[int, str]).warnings == ("unsupported_mapping_key",)
    assert spec_for(list[bool]).warnings == ("unsupported_list_item",)
    assert spec_for(list[list[str]]).engine == "skip"


def test_uuid_and_urls_are_text() -> None:
    import uuid

    from pydantic import AnyUrl

    assert clairic_dict(spec_for(uuid.UUID))["type"] == "text"
    assert clairic_dict(spec_for(AnyUrl))["type"] == "text"


def test_max_depth_guard_skips_nested_models() -> None:
    class Outer(BaseModel):
        pet: Pet
        pets: list[Pet]
        name: str

    default = compile_schema(Outer)
    assert default.spec(("pet",)).engine == "submodel"
    shallow = compile_schema(Outer, limits=Limits(max_depth=1))
    assert shallow.spec(("pet",)).engine == "skip"
    assert shallow.spec(("pet",)).warnings == ("max_depth",)
    assert shallow.spec(("pets",)).warnings == ("max_depth",)
    assert shallow.spec(("name",)).engine == "clairic"
    assert any("max_depth" in message for message in shallow.warnings)


def test_recursive_models_expand_down_to_the_depth_limit() -> None:
    class Node(BaseModel):
        name: str
        parent: Optional["Node"] = None
        children: list["Node"] = []

    Node.model_rebuild()
    plan = compile_schema(Node, limits=Limits(max_depth=3))
    assert plan.spec(("parent",)).engine == "submodel"
    assert plan.spec(("parent", "parent")).engine == "submodel"
    # leaves are at most 3 deep: the third `parent` would put `name` at depth 4
    assert plan.spec(("parent", "parent", "parent")).engine == "skip"
    assert plan.spec(("parent", "parent", "parent")).warnings == ("max_depth",)
    # depth first in declaration order; the collection of the deepest level is
    # past the limit and skipped
    assert [spec.pretty for spec in plan.collection_specs()] == [
        "parent.children",
        "children",
    ]
    assert plan.spec(("parent", "parent", "children")).warnings == ("max_depth",)
    shallow = compile_schema(Node, limits=Limits(max_depth=1))
    assert shallow.spec(("parent",)).warnings == ("max_depth",)


# --------------------------------------------------------------------------- #
# Annotation forms
# --------------------------------------------------------------------------- #

UserId = NewType("UserId", int)
Age = TypeAliasType("Age", int)


@pytest.mark.parametrize(
    ("annotation", "kind"),
    [
        (Annotated[int, Field(ge=0)], "number"),
        (typing_extensions.Annotated[str, "meta"], "text"),
        (Optional[Annotated[int, Field(gt=0)]], "number"),
        (int | None, "number"),
        (Union[str, None], "text"),
        (UserId, "number"),
        (Age, "number"),
        (Annotated[Annotated[dt.date, "x"], "y"], "date"),
        (list[Annotated[str, Field(min_length=2)]], "list"),
        (typing_extensions.Literal["a", "b"], "choice"),
    ],
)
def test_annotation_forms(annotation: Any, kind: str) -> None:
    spec = spec_for(annotation)
    if kind == "choice":
        assert clef_dict(spec)["type"] == "choice"
    else:
        assert clairic_dict(spec)["type"] == kind


def test_annotated_constraints_reach_the_prose() -> None:
    spec = spec_for(Optional[Annotated[int, Field(gt=0)]])
    assert spec.instructions == "Thing. Must be greater than 0."


def test_literal_none_makes_a_field_optional() -> None:
    spec = spec_for(Literal["a", "b", None])
    assert list(spec.key_map or {}) == ["a", "b", "unstated"]
    assert spec.required is False


# --------------------------------------------------------------------------- #
# Required / optional and thresholds
# --------------------------------------------------------------------------- #


def test_required_field_uses_the_required_threshold() -> None:
    spec = spec_for(str)
    assert spec.required is True
    assert spec.has_default is False
    assert spec.null_threshold == REQUIRED_THRESHOLD
    assert clairic_dict(spec)["null_threshold"] == REQUIRED_THRESHOLD


@pytest.mark.parametrize(
    ("annotation", "default", "has_default"),
    [
        (Optional[str], ..., False),
        (str | None, None, True),
        (str, "x", True),
    ],
)
def test_optional_or_defaulted_fields_use_the_optional_threshold(
    annotation: Any, default: Any, has_default: bool
) -> None:
    spec = spec_for(annotation, default)
    assert spec.required is False
    assert spec.has_default is has_default
    assert spec.null_threshold == OPTIONAL_THRESHOLD
    assert clairic_dict(spec)["null_threshold"] == OPTIONAL_THRESHOLD


def test_default_factory_counts_as_a_default() -> None:
    model = create_model(
        "M", f=(list[str], Field(default_factory=list, description=DESC))
    )
    spec = compile_schema(model).specs[0]
    assert spec.has_default and not spec.required


def test_explicit_null_threshold_wins() -> None:
    spec = spec_for(str, json_schema_extra=s1_field(null_threshold=0.9))
    assert spec.null_threshold == 0.9
    assert clairic_dict(spec)["null_threshold"] == 0.9
    spec = spec_for(str, "x", json_schema_extra=s1_field(null_threshold=0.1))
    assert spec.null_threshold == 0.1


def test_gates_set_thresholds_and_confidence_floors() -> None:
    fields: dict[str, Any] = {
        "a": (str, Field(..., description="A")),
        "b": (Optional[str], Field(None, description="B")),
        "c": (Literal["x", "y"], Field(..., description="C")),
        "d": (
            str,
            Field(..., description="D", json_schema_extra=s1_field(min_confidence=0.9)),
        ),
    }
    model = create_model("M", **fields)
    gates = GateConfig(
        required_null_threshold=0.8,
        optional_null_threshold=0.2,
        min_confidence=0.4,
        min_choice_confidence=0.6,
    )
    plan = compile_schema(model, gates=gates)
    assert [spec.null_threshold for spec in plan.specs] == [0.8, 0.2, 0.8, 0.8]
    assert [spec.min_confidence for spec in plan.specs] == [0.4, 0.4, 0.6, 0.9]


def test_bounds_are_prose_not_clairic_min_max() -> None:
    spec = spec_for(Annotated[int, Field(ge=0, le=150)])
    payload = clairic_dict(spec)
    assert "min" not in payload and "max" not in payload
    assert payload["instructions"] == "Thing. Must be at least 0 and at most 150."


# --------------------------------------------------------------------------- #
# Instruction synthesis
# --------------------------------------------------------------------------- #


class Documented(BaseModel):
    """A documented model."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    described: Annotated[str, Field(description="From description", title="From title")]
    titled: Annotated[str, Field(title="From title")]
    docstring: str
    """From the attribute docstring."""
    due_date: dt.date
    dueDateTime: dt.datetime


def test_description_precedence() -> None:
    plan = compile_schema(Documented)
    by_name = {spec.pretty: spec.instructions for spec in plan.specs}
    assert by_name == {
        "described": "From description",
        "titled": "From title",
        "docstring": "From the attribute docstring.",
        "due_date": "due date",
        "dueDateTime": "due date time",
    }
    assert plan.description == "A documented model."


def test_description_comes_before_type_and_constraint_hints() -> None:
    spec = spec_for(
        Annotated[str, Field(min_length=3)],
        description="Name of the thing",
    )
    assert spec.instructions == "Name of the thing. At least 3 characters."
    assert spec.description == "Name of the thing"


def test_s1_instructions_replace_the_description() -> None:
    spec = spec_for(
        Annotated[str, Field(max_length=9)],
        json_schema_extra=s1_field(instructions="The legal name"),
    )
    assert spec.instructions == "The legal name. At most 9 characters."


@pytest.mark.parametrize(
    ("annotation", "kwargs", "expected"),
    [
        (Annotated[int, Field(ge=1)], {}, "Thing. Must be at least 1."),
        (Annotated[int, Field(gt=1)], {}, "Thing. Must be greater than 1."),
        (Annotated[int, Field(le=5)], {}, "Thing. Must be at most 5."),
        (Annotated[int, Field(lt=5)], {}, "Thing. Must be less than 5."),
        (
            Annotated[int, Field(gt=1, lt=5)],
            {},
            "Thing. Must be greater than 1 and less than 5.",
        ),
        (
            Annotated[int, Field(multiple_of=5)],
            {},
            "Thing. Must be a multiple of 5.",
        ),
        (
            Annotated[Decimal, Field(max_digits=6, decimal_places=2, ge=0)],
            {},
            "Thing. Must be at least 0. At most 6 digits in total.",
        ),
        (
            Annotated[dt.date, Field(ge=dt.date(2020, 1, 1), lt=dt.date(2030, 1, 1))],
            {},
            "Thing. Must be on or after 2020-01-01 and before 2030-01-01.",
        ),
        (Annotated[str, Field(max_length=3)], {}, "Thing. At most 3 characters."),
        (
            Annotated[str, Field(pattern=r"^\d+$")],
            {},
            r"Thing. Must match the pattern ^\d+$.",
        ),
        (Annotated[list[str], Field(min_length=1)], {}, "Thing. At least 1 items."),
        (
            Annotated[list[str], Field(min_length=1, max_length=4)],
            {},
            "Thing. Between 1 and 4 items.",
        ),
    ],
)
def test_constraint_prose(
    annotation: Any, kwargs: dict[str, Any], expected: str
) -> None:
    assert spec_for(annotation, **kwargs).instructions == expected


def test_instructions_are_clipped_to_the_limit() -> None:
    model = create_model("M", f=(str, Field(..., description="word " * 200)))
    plan = compile_schema(model, limits=Limits(max_instruction_chars=60))
    instructions = plan.specs[0].instructions
    assert len(instructions) <= 60
    assert instructions.endswith("...")
    assert clairic_dict(plan.specs[0])["instructions"] == instructions
    default = compile_schema(model)
    assert len(default.specs[0].instructions) <= 600


def test_clef_questions_are_clipped_too() -> None:
    model = create_model("M", f=(bool, Field(..., description="word " * 200)))
    plan = compile_schema(model, limits=Limits(max_instruction_chars=80))
    assert len(clef_dict(plan.specs[0])["instructions"]) <= 80


def test_shortened_names_get_the_parent_context() -> None:
    long = "x" * 80
    fields: dict[str, Any] = {
        long: (str, Field(..., description="Shortened one")),
        "short": (str, Field(..., description="Plain one")),
        "y" * 105: (bool, Field(..., description="Shortened flag")),
    }
    model = create_model("Long", **fields)
    plan = compile_schema(model)
    shortened = plan.clairic_names.shortened | plan.clef_names.shortened
    assert len(shortened) == 2
    for spec in plan.specs:
        if spec.wire_name in shortened:
            context = "Long"
            assert context  # the context is never empty
            assert spec.instructions.startswith(f"{context}: ")
            assert len(spec.wire_name) <= 64 or spec.clef_name == spec.wire_name
        else:
            assert spec.instructions == "Plain one"
    clef = plan.clef_questions()
    (question,) = clef.values()
    assert isinstance(question.instructions, str)
    assert question.instructions.startswith("Long: ")


def test_shortened_context_uses_the_model_title() -> None:
    fields: dict[str, Any] = {"z" * 90: (str, Field(..., description="Deep"))}
    model = create_model(
        "Customer", __config__=ConfigDict(title="Customer record"), **fields
    )
    plan = compile_schema(model)
    assert plan.title == "Customer record"
    assert plan.specs[0].instructions == "Customer record: Deep"


def test_aggregation_lint_warns_and_recommends_computed_field() -> None:
    class Order(BaseModel):
        total: int = Field(description="Total of all line items")
        mean: float = Field(description="The Mean of the scores")
        n: int = Field(description="Count of orders")
        agg: int = Field(description="An AGGREGATE figure")
        fine: int = Field(description="The totals line, a count")
        word: str = Field(description="Accounting average")

    plan = compile_schema(Order)
    flagged = [spec.pretty for spec in plan.specs if "aggregation" in spec.warnings]
    assert flagged == ["total", "mean", "n", "agg"]
    assert len(plan.warnings) == 4
    assert all("computed_field" in message for message in plan.warnings)
    assert plan.warnings[0].startswith("total:")


def test_computed_fields_are_not_asked_for() -> None:
    class Order(BaseModel):
        a: int
        b: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def total(self) -> int:
            return self.a + self.b

    assert [spec.pretty for spec in compile_schema(Order).specs] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def test_identity_marker_wins_and_may_be_compound() -> None:
    class Rec(BaseModel):
        name: str
        code: Annotated[str, Field(json_schema_extra=s1_field(identity=True))]
        zone: Annotated[str, Field(json_schema_extra=s1_field(identity=True))]

    plan = compile_schema(Rec)
    assert plan.identity == ("code", "zone")
    assert plan.spec(("code",)).identity is True
    assert plan.spec(("name",)).identity is False


def test_identity_prefers_well_known_names_in_priority_order() -> None:
    class Rec(BaseModel):
        email: str
        title: str
        sku: str
        notes: str

    assert compile_schema(Rec).identity == ("title",)

    class Rec2(BaseModel):
        notes: str
        email: str
        id: str

    assert compile_schema(Rec2).identity == ("id",)


def test_identity_well_known_name_must_be_a_str_field() -> None:
    class Rec(BaseModel):
        name: int
        title: str | None = None
        other: str

    assert compile_schema(Rec).identity == ("title",)


def test_identity_falls_back_to_the_first_required_str_field() -> None:
    class Rec(BaseModel):
        count: int
        optional: str | None = None
        company: str
        city: str

    assert compile_schema(Rec).identity == ("company",)


def test_identity_is_empty_without_candidates() -> None:
    class Rec(BaseModel):
        count: int
        note: str | None = None
        kind: Literal["a", "b"]

    assert compile_schema(Rec).identity == ()


# --------------------------------------------------------------------------- #
# Plan structure, caching and inputs
# --------------------------------------------------------------------------- #


class Everything(BaseModel):
    """Every supported flat type."""

    text_field: str
    integer: int
    number: float
    money: Decimal
    day: dt.date
    clock: dt.time
    moment: dt.datetime
    span: dt.timedelta
    words: list[str]
    counts: list[int]
    values: list[float]
    days: list[dt.date]
    uniq: set[int]
    frozen: frozenset[str]
    moments: list[dt.datetime]
    times: list[dt.time]
    spans: list[dt.timedelta]
    picks: list[Literal["a", "b"]]
    colors: list[Color]
    flag: bool
    maybe: bool | None = None
    mode: Literal["x", "y"]
    color: Color
    size: Size
    rank: Rank
    impact: Annotated[Impact, Field(json_schema_extra=s1_field(clef="score"))]
    either: int | str
    bounded: Annotated[int, Field(ge=0, le=10)]
    named: Annotated[str, Field(min_length=1, pattern="^[A-Z]")]


def test_every_flat_type_builds_valid_requests() -> None:
    plan = compile_schema(Everything)
    fields = plan.clairic_fields()
    questions = plan.clef_questions()
    assert len(plan.specs) == len(Everything.model_fields)
    assert plan.collection_specs() == ()
    assert all(spec.engine in ("clairic", "clef") for spec in plan.specs)
    clairic = ClairicRequest(state="Some text.", fields=fields)
    clef = ClefRequest(model="clef", state="Some text.", questions=questions)
    # to_payload() is checked against the published JSON Schemas by conftest.
    assert set(clairic.to_payload()["fields"]) == set(fields)
    assert set(clef.to_payload()["questions"]) == set(questions)
    assert "either.__type" in questions
    assert "flag" in questions and "flag" not in fields
    assert "text_field" in fields


def test_wire_names_are_unique_per_namespace() -> None:
    plan = compile_schema(Everything)
    assert len(plan.clairic_names) == len(plan.clairic_fields())
    assert len(plan.clef_names) == len(plan.clef_questions())
    for spec in plan.specs:
        if spec.clairic is not None:
            assert plan.clairic_names.path_to_name[spec.path] == spec.wire_name
        if spec.clef is not None:
            assert plan.clef_names.name_to_path[spec.clef_name]


def test_plan_describes_the_model() -> None:
    plan = compile_schema(Everything)
    assert plan.schema is Everything
    assert plan.name == "Everything"
    assert plan.title == "Everything"
    assert plan.description == "Every supported flat type."
    assert isinstance(plan, SchemaPlan)
    assert plan.spec(("integer",)).annotation is int
    assert plan.spec(("integer",)).pretty == "integer"
    with pytest.raises(KeyError):
        plan.spec(("missing",))


def test_paths_select_a_subset_in_declaration_order() -> None:
    plan = compile_schema(Everything)
    paths = [("flag",), ("integer",), ("either",), ("text_field",)]
    assert list(plan.clairic_fields(paths)) == ["text_field", "integer", "either"]
    assert list(plan.clef_questions(paths)) == ["flag", "either.__type"]
    assert plan.clairic_fields([]) == {}
    with pytest.raises(KeyError):
        plan.clairic_fields([("nope",)])


def test_specs_are_frozen_and_hashable() -> None:
    spec = compile_schema(Everything).spec(("mode",))
    assert hash(spec) == hash(compile_schema(Everything).spec(("mode",)))
    with pytest.raises(AttributeError):
        spec.required = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.key_map["x"] = 1  # type: ignore[index]
    assert {spec: 1}[spec] == 1


def test_compile_is_memoised_on_model_gates_and_limits() -> None:
    first = compile_schema(Everything)
    assert compile_schema(Everything) is first
    assert compile_schema(Everything, gates=GateConfig()) is first
    assert compile_schema(Everything, limits=Limits()) is first
    other = compile_schema(Everything, gates=GateConfig(noul_margin=0.2))
    assert other is not first
    assert compile_schema(Everything, gates=GateConfig(noul_margin=0.2)) is other
    assert compile_schema(Everything, limits=Limits(max_depth=2)) is not first


def test_typed_dict_and_function_inputs_compile_and_cache() -> None:
    class Address(TypedDict):
        """An address."""

        street: str
        zip_code: NotRequired[str]

    plan = compile_schema(Address)
    assert plan.name == "Address"
    assert plan.description == "An address."
    assert plan.spec(("zip_code",)).required is False
    assert plan.spec(("zip_code",)).has_default is True
    assert plan.identity == ("street",)
    assert compile_schema(Address) is plan

    def lookup(city: str, limit: int = 5) -> None:
        """Find things.

        Args:
            city: City to search.
        """

    plan = compile_schema(lookup)
    assert plan.name == "lookup"
    assert plan.spec(("city",)).instructions == "City to search."
    assert plan.spec(("limit",)).required is False
    assert compile_schema(lookup) is plan


def test_json_schema_input_compiles() -> None:
    pytest.importorskip("dydantic")
    plan = compile_schema(
        {
            "title": "Ticket",
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "One-line summary"},
                "points": {"type": "integer"},
            },
            "required": ["summary"],
        }
    )
    assert plan.name == "Ticket"
    assert plan.spec(("summary",)).instructions == "One-line summary"
    assert plan.spec(("summary",)).required is True
    assert plan.spec(("points",)).required is False
    assert clairic_dict(plan.spec(("points",)))["integer"] is True


def test_unknown_s1_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        spec_for(str, json_schema_extra={"trustcall_system_one": {"nope": 1}})


def test_extra_options_are_recorded_on_the_spec() -> None:
    spec = spec_for(str, json_schema_extra=s1_field(unit="kg", min_confidence=0.8))
    assert dict(spec.extra) == {"unit": "kg", "min_confidence": 0.8}
    assert dict(spec_for(str).extra) == {}


# --------------------------------------------------------------------------- #
# s1_field
# --------------------------------------------------------------------------- #


def test_s1_field_returns_only_the_options_that_were_set() -> None:
    assert s1_field() == {"trustcall_system_one": {}}
    assert s1_field(unit="hours", identity=True) == {
        "trustcall_system_one": {"unit": "hours", "identity": True}
    }
    assert s1_field(identity=False, skip=False) == {"trustcall_system_one": {}}
    assert s1_field(integer=False, decimals=0) == {
        "trustcall_system_one": {"integer": False, "decimals": 0}
    }
    everything = s1_field(
        instructions="i",
        null_threshold=0.5,
        min_confidence=0.6,
        clairic_type="text",
        unit="u",
        decimals=2,
        integer=True,
        identity=True,
        skip=True,
        clef="choice",
        criteria={"a": "b"},
    )
    assert set(everything["trustcall_system_one"]) == {
        "instructions",
        "null_threshold",
        "min_confidence",
        "clairic_type",
        "unit",
        "decimals",
        "integer",
        "identity",
        "skip",
        "clef",
        "criteria",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clef": "maybe"},
        {"null_threshold": 0},
        {"null_threshold": 1.5},
        {"null_threshold": True},
        {"min_confidence": 0},
        {"min_confidence": 2},
        {"clairic_type": "blob"},
        {"decimals": 7},
        {"decimals": -1},
        {"decimals": 1.5},
        {"integer": "yes"},
        {"instructions": "  "},
        {"unit": ""},
        {"criteria": "text"},
    ],
)
def test_s1_field_validates_its_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        s1_field(**kwargs)


def test_s1_field_null_threshold_may_be_one() -> None:
    assert s1_field(null_threshold=1)["trustcall_system_one"]["null_threshold"] == 1


def test_s1_field_works_inside_field() -> None:
    class Rec(BaseModel):
        name: str = Field(json_schema_extra=s1_field(identity=True))

    assert Rec.model_fields["name"].json_schema_extra == {
        "trustcall_system_one": {"identity": True}
    }
    assert "trustcall_system_one" in Rec.model_json_schema()["properties"]["name"]


def test_context_helper_is_never_empty_and_joins_ancestors() -> None:
    from trustcall_system_one._compile import _context

    assert _context("Person", ("name",)) == "Person"
    assert _context("Person", ("address", "street")) == "Person → address"
    assert _context("Person", ("work_address", 0, "geo", "lat")) == (
        "Person → work address → geo"
    )
    assert _context("T", ("a", "b", "c")) == "T → a → b"


# --------------------------------------------------------------------------- #
# Review fixes: unstated options, frozen wire models, aliases, types, names
# --------------------------------------------------------------------------- #


def option_ids(spec: FieldSpec) -> list[str]:
    return list(spec.key_map or {})


@pytest.mark.parametrize(
    ("annotation", "default", "expected"),
    [
        (Optional[Literal["open", "closed"]], None, ["open", "closed", "unstated"]),
        (Optional[Color], None, ["RED", "BLUE", "unstated"]),
        (Color | None, ..., ["RED", "BLUE", "unstated"]),
        (Color, Color.RED, ["RED", "BLUE", "unstated"]),
        (Literal["open", "closed"], "open", ["open", "closed", "unstated"]),
        (Optional[bool], None, ["yes", "no", "unstated"]),
    ],
)
def test_nullable_or_defaulted_choices_get_an_unstated_option(
    annotation: Any, default: Any, expected: list[str]
) -> None:
    spec = spec_for(annotation, default)
    assert option_ids(spec) == expected
    assert (spec.key_map or {})["unstated"] is UNSTATED
    assert clef_dict(spec)["criteria"]["unstated"] == "The input does not state this."
    assert spec.null_threshold == OPTIONAL_THRESHOLD


@pytest.mark.parametrize("annotation", [Literal["open", "closed"], Color, bool])
def test_required_choices_get_no_unstated_option(annotation: Any) -> None:
    spec = spec_for(annotation, json_schema_extra=s1_field(clef="choice"))
    assert "unstated" not in option_ids(spec)
    assert UNSTATED not in (spec.key_map or {}).values()


def test_required_bool_forced_to_a_choice_is_just_yes_no() -> None:
    spec = spec_for(bool, json_schema_extra=s1_field(clef="choice"))
    assert option_ids(spec) == ["yes", "no"]
    assert dict(spec.key_map or {}) == {"yes": True, "no": False}


def test_defaulted_bool_forced_to_a_choice_gets_unstated() -> None:
    spec = spec_for(bool, False, json_schema_extra=s1_field(clef="choice"))
    assert option_ids(spec) == ["yes", "no", "unstated"]
    assert spec.nullable is False


def test_unstated_id_is_renamed_when_it_collides() -> None:
    spec = spec_for(Optional[Literal["unstated", "a", "unstated_2"]], None)
    assert option_ids(spec) == ["unstated", "a", "unstated_2", "unstated_3"]
    assert (spec.key_map or {})["unstated_3"] is UNSTATED
    assert (spec.key_map or {})["unstated"] == "unstated"


def test_unstated_counts_towards_the_255_option_limit() -> None:
    assert len(spec_for(make_enum("Fits", 254), None).key_map or {}) == 255
    with pytest.raises(ValueError, match=r"'f'.*256 options"):
        spec_for(make_enum("Over", 255), None)


def test_scores_never_get_an_unstated_option() -> None:
    spec = spec_for(Rank, Rank.LOW, json_schema_extra=s1_field(clef="score"))
    assert dict(spec.key_map or {}) == {"0": 1, "1": 2}


def test_unstated_sentinel_survives_copying_and_pickling() -> None:
    import copy
    import pickle

    assert copy.deepcopy(UNSTATED) is UNSTATED
    assert copy.copy(UNSTATED) is UNSTATED
    assert pickle.loads(pickle.dumps(UNSTATED)) is UNSTATED
    assert repr(UNSTATED) == "UNSTATED"
    spec = spec_for(Optional[Color], None)
    assert spec.key_map is not None and hash(spec) == hash(spec)


def test_wire_models_are_frozen() -> None:
    plan = compile_schema(Everything)
    field = plan.clairic_fields()["integer"]
    with pytest.raises(ValidationError):
        field.instructions = "changed"  # type: ignore[misc]
    question = plan.clef_questions()["flag"]
    with pytest.raises(ValidationError):
        question.instructions = "changed"  # type: ignore[misc]
    request = ClairicRequest(state="x", fields={"a": field})
    with pytest.raises(ValidationError):
        request.state = "y"  # type: ignore[misc]


def test_mutating_returned_objects_does_not_leak_into_the_cached_plan() -> None:
    class Leaky(BaseModel):
        kind: Literal["a", "b"]
        n: int

    plan = compile_schema(Leaky)
    fields = plan.clairic_fields()
    fields["extra"] = fields["n"]
    fields.pop("n")
    questions = plan.clef_questions()
    question = questions["kind"]
    assert isinstance(question, ChoiceQuestion)
    question.criteria["a"] = "tampered"
    question.criteria["zzz"] = "tampered"
    questions.clear()
    again = compile_schema(Leaky)
    assert list(again.clairic_fields()) == ["n"]
    fresh = again.clef_questions()["kind"]
    assert isinstance(fresh, ChoiceQuestion)
    assert fresh.criteria == {"a": "a", "b": "b"}
    assert fresh is not question


def test_alias_free_fields_have_no_alias() -> None:
    assert spec_for(int).alias is None


def test_aliases_are_recorded_for_validation() -> None:
    from pydantic import AliasChoices, AliasPath

    class Aliased(BaseModel):
        plain: str
        by_alias: str = Field(alias="byAlias")
        by_validation: str = Field(validation_alias="byValidation")
        both: str = Field(alias="a", validation_alias="v")
        choices: str = Field(validation_alias=AliasChoices(AliasPath("x", 0), "second"))
        path: str = Field(validation_alias=AliasPath("x", 0))
        same: str = Field(alias="same")

    plan = compile_schema(Aliased)
    aliases = {spec.pretty: spec.alias for spec in plan.specs}
    assert aliases == {
        "plain": None,
        "by_alias": "byAlias",
        "by_validation": "byValidation",
        "both": "v",
        "choices": "second",
        "path": None,
        "same": None,
    }
    assert plan.spec(("path",)).warnings == ("alias_path",)
    assert plan.spec(("choices",)).warnings == ()
    assert any("AliasPath" in message for message in plan.warnings)
    assert plan.spec(("by_alias",)).wire_name == "by_alias"


# Pydantic and stdlib types ---------------------------------------------------


def test_network_and_text_like_types_are_text() -> None:
    import ipaddress

    from pydantic import HttpUrl, IPvAnyAddress, IPvAnyInterface, IPvAnyNetwork

    for annotation in (
        HttpUrl,
        IPvAnyAddress,
        IPvAnyInterface,
        IPvAnyNetwork,
        ipaddress.IPv4Address,
        Optional[HttpUrl],
    ):
        spec = spec_for(annotation)
        assert clairic_dict(spec)["type"] == "text", annotation
        assert spec.warnings == ()


def test_email_types_are_text() -> None:
    pytest.importorskip("email_validator")
    from pydantic import EmailStr, NameEmail

    assert clairic_dict(spec_for(EmailStr))["type"] == "text"
    assert clairic_dict(spec_for(NameEmail))["type"] == "text"


def test_email_types_are_recognised_without_email_validator() -> None:
    from trustcall_system_one._compile import _TEXT_LIKE

    names = {cls.__name__ for cls in _TEXT_LIKE}
    assert {"EmailStr", "NameEmail", "IPvAnyAddress"} <= names


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("PastDate", "date"),
        ("FutureDate", "date"),
        ("AwareDatetime", "datetime"),
        ("NaiveDatetime", "datetime"),
        ("PastDatetime", "datetime"),
        ("FutureDatetime", "datetime"),
    ],
)
def test_pydantic_temporal_types_use_their_base_type(name: str, kind: str) -> None:
    import pydantic

    annotation: Any = getattr(pydantic, name)
    assert clairic_dict(spec_for(annotation))["type"] == kind
    assert clairic_dict(spec_for(Optional[annotation]))["type"] == kind
    listed = spec_for(list[annotation])
    assert clairic_dict(listed)["type"] == "list"
    assert listed.item_kind == kind


def test_variadic_tuples_are_lists() -> None:
    spec = spec_for(tuple[str, ...])
    assert clairic_dict(spec) == numeric("list", items="text")
    assert spec.item_kind == "str"
    assert spec.post == ""
    assert clairic_dict(spec_for(tuple[int, ...]))["items"] == "number"
    assert clairic_dict(spec_for(tuple[dt.date, ...]))["items"] == "date"
    assert spec_for(tuple[dt.datetime, ...]).warnings == ("lossy_list",)
    assert spec_for(tuple[int, str]).warnings == ("unsupported_type",)
    assert spec_for(tuple).warnings == ("unsupported_type",)


def test_nested_typed_dicts_compile_as_submodels_and_collections() -> None:
    class Geo(TypedDict):
        lat: float

    class Place(TypedDict):
        name: str
        geo: Geo
        spots: list[Geo]
        maybe: Optional[Geo]
        later: NotRequired[Geo]

    plan = compile_schema(Place)
    engines = {spec.pretty: spec.engine for spec in plan.specs}
    assert engines == {
        "name": "clairic",
        "geo": "submodel",
        "geo.lat": "clairic",
        "spots": "collection",
        "maybe": "submodel",
        "maybe.lat": "clairic",
        "later": "submodel",
        "later.lat": "clairic",
    }
    geo = plan.spec(("geo",)).annotation
    assert issubclass(geo, BaseModel) and geo.__name__ == "Geo"
    assert geo.model_fields["lat"].is_required()
    assert plan.spec(("later",)).has_default is True


def test_typed_dict_fields_inside_a_model_are_recorded_for_the_expander() -> None:
    class Geo(TypedDict):
        lat: float

    class Owner(BaseModel):
        home: Optional[Geo] = None
        homes: list[Geo] = []

    plan = compile_schema(Owner)
    assert plan.spec(("home",)).engine == "submodel"
    assert plan.spec(("home", "lat")).engine == "clairic"
    assert plan.spec(("homes",)).engine == "collection"


def test_function_parameters_may_be_typed_dicts() -> None:
    class Geo(TypedDict):
        lat: float

    def where(point: Geo, trail: list[Geo]) -> None: ...

    plan = compile_schema(where)
    assert [spec.engine for spec in plan.specs] == [
        "submodel",
        "clairic",
        "collection",
    ]


class Tree(TypedDict):
    kids: list["Tree"]


def test_recursive_typed_dicts_are_rejected() -> None:
    with pytest.raises(TypeError, match="recursive"):
        compile_schema(Tree)


def test_dict_of_models_stays_a_mapping() -> None:
    spec = spec_for(dict[str, Pet])
    assert spec.engine == "mapping"
    assert spec.warnings == ()
    assert spec.clairic is None and spec.clef is None
    assert spec_for(Optional[dict[str, Pet]]).engine == "mapping"


# Option ids and criteria ------------------------------------------------------


def test_int_enum_options_use_the_member_name_and_keep_the_value() -> None:
    spec = spec_for(Rank)
    assert clef_dict(spec)["criteria"] == {"LOW": "LOW (1)", "HIGH": "HIGH (2)"}
    assert dict(spec.key_map or {}) == {"LOW": 1, "HIGH": 2}


def test_literal_option_ids_keep_the_original_spelling_in_the_criteria() -> None:
    spec = spec_for(Literal["N/A", "<18", "in progress", "done"])
    assert clef_dict(spec)["criteria"] == {
        "N_A": "N/A",
        "_18": "<18",
        "in_progress": "in progress",
        "done": "done",
    }
    assert dict(spec.key_map or {}) == {
        "N_A": "N/A",
        "_18": "<18",
        "in_progress": "in progress",
        "done": "done",
    }


def test_enum_member_names_are_deduplicated_after_sanitising() -> None:
    factory: Any = enum.Enum
    spec = spec_for(factory("Odd", {"a b": 1, "a_b": 2, "A": 3}))
    assert option_ids(spec) == ["a_b", "a_b_2", "A"]


def test_enum_whose_name_equals_its_value_needs_no_label() -> None:
    class Plain(enum.Enum):
        YES = "YES"
        NO = "NO"

    assert clef_dict(spec_for(Plain))["criteria"] == {"YES": "YES", "NO": "NO"}


# Names ----------------------------------------------------------------------------


def test_colliding_plain_names_all_get_the_digest_form_in_any_order() -> None:
    fields: dict[str, Any] = {
        "a b": (str, Field(..., description="one")),
        "a_b": (str, Field(..., description="two")),
        "other": (str, Field(..., description="three")),
    }
    forward = compile_schema(create_model("Fwd", **fields))
    backward = compile_schema(
        create_model("Bwd", **dict(reversed(list(fields.items()))))
    )
    for plan in (forward, backward):
        names = {spec.pretty: spec.wire_name for spec in plan.specs}
        assert names["other"] == "other"
        assert names["a b"].startswith("a_b_a_b_")
        assert names["a_b"].startswith("a_b_a_b_")
        assert names["a b"] != names["a_b"]
        assert len(plan.clairic_names.shortened) == 2
    assert {s.pretty: s.wire_name for s in forward.specs} == {
        s.pretty: s.wire_name for s in backward.specs
    }


# Descriptions and overrides --------------------------------------------------


def test_doc_annotation_is_a_description_source() -> None:
    from typing_extensions import Doc

    spec = spec_for(Annotated[int, Doc("From Doc")], description=None, title="T")
    assert spec.description == "From Doc"
    spec = spec_for(Annotated[int, Doc("From Doc")], description="From Field")
    assert spec.description == "From Field"
    spec = spec_for(int, description=None, title="From title")
    assert spec.description == "From title"
    spec = spec_for(Optional[Annotated[int, Doc("Inner")]], description=None)
    assert spec.description == "Inner"


@pytest.mark.parametrize(
    ("annotation", "override", "warns"),
    [
        (str, "date", True),
        (str, "number", True),
        (int, "text", True),
        (dt.date, "datetime", True),
        (str, "text", False),
        (str, "list", True),
    ],
)
def test_mismatching_clairic_type_overrides_warn(
    annotation: Any, override: str, warns: bool
) -> None:
    kwargs: dict[str, Any] = {"clairic_type": override}
    spec = spec_for(annotation, json_schema_extra=s1_field(**kwargs))
    assert ("clairic_type_mismatch" in spec.warnings) is warns
    if warns:
        assert any(
            "clairic_type" in message
            for message in compile_schema(
                create_model(
                    "W",
                    f=(annotation, Field(..., json_schema_extra=s1_field(**kwargs))),
                )
            ).warnings
        )


def test_numeric_overrides_between_number_kinds_do_not_warn() -> None:
    model = create_model(
        "M",
        f=(
            Annotated[float, Field(ge=0, le=1)],
            Field(..., json_schema_extra=s1_field(clairic_type="estimate")),
        ),
    )
    assert compile_schema(model).specs[0].warnings == ()


def test_estimate_override_takes_min_and_max_from_the_bounds() -> None:
    model = create_model(
        "M",
        f=(
            Annotated[float, Field(ge=0, lt=1)],
            Field(..., json_schema_extra=s1_field(clairic_type="forecast")),
        ),
    )
    payload = clairic_dict(compile_schema(model).specs[0])
    assert (payload["type"], payload["min"], payload["max"]) == ("forecast", 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Instruction parts
# --------------------------------------------------------------------------- #


def test_render_instructions_builds_every_role() -> None:
    from trustcall_system_one._compile import render_instructions

    assert render_instructions("clairic", "the age") == "the age"
    assert (
        render_instructions("clairic", "the age", prefix="P", hints=["At least 0."])
        == "P: the age. At least 0."
    )
    assert render_instructions("noul", "x") == "Does the input say this is true: x?"
    assert render_instructions("choice", "x") == "Pick the option that best fits: x"
    assert render_instructions("score", "x").startswith("Rate on the scale")
    assert render_instructions("union_type", "x") == "What kind of value is this: x?"
    clipped = render_instructions("choice", "word " * 50, limit=40)
    assert len(clipped) <= 40 and clipped.endswith("...")


def test_specs_keep_the_parts_the_instructions_were_built_from() -> None:
    from trustcall_system_one._compile import render_instructions

    spec = spec_for(int, ge=3)
    assert spec.hints == ("Must be at least 3.",)
    assert spec.context == "" and spec.clef_context == ""
    assert (
        render_instructions(
            "clairic", spec.description, prefix=spec.context, hints=spec.hints
        )
        == spec.instructions
    )
    assert spec_for(bool).hints == ()
    union = spec_for(Union[int, str])
    assert union.hints == () and union.clef_context == ""


def test_shortened_names_record_their_context() -> None:
    long_name = "a_really_long_field_name_" * 4
    fields: dict[str, Any] = {long_name: (Optional[int], None)}
    model = create_model("Ctx", **fields)
    spec = compile_schema(model).specs[0]
    assert spec.context == "Ctx"
    assert spec.instructions.startswith("Ctx: ")


# --------------------------------------------------------------------------- #
# Class docstrings of nested models
# --------------------------------------------------------------------------- #


class HhMm(BaseModel):
    """A duration as printed on a sheet. Hours and minutes are two parts; the total
    in minutes is hours times 60 plus the minutes (18542 hours is 1112520 minutes)."""

    raw: Optional[str] = None
    total_minutes: Optional[int] = None


class Airframe(BaseModel):
    """Airframe usage."""

    total_airframe_hours: Optional[HhMm] = None
    cycles: Optional[int] = Field(None, description="landing cycles")


class Plane(BaseModel):
    """An aircraft."""

    airframe: Airframe = Field(default_factory=Airframe)


def test_a_nested_models_docstring_reaches_the_instructions_of_its_leaves() -> None:
    spec = compile_schema(Plane).spec(
        ("airframe", "total_airframe_hours", "total_minutes")
    )
    assert "hours times 60 plus the minutes" in spec.instructions
    assert spec.context.startswith("Plane → airframe → total airframe hours (")
    assert spec.instructions.startswith(spec.context + ": total minutes")
    assert len(spec.instructions) <= Limits().max_instruction_chars
    # the engine rebuilds the text from the parts: it must say the same
    assert (
        render_instructions(
            "clairic", spec.description, prefix=spec.context, hints=spec.hints
        )
        == spec.instructions
    )


def test_a_field_without_a_description_is_described_by_its_models_docstring() -> None:
    plan = compile_schema(Plane)
    group = plan.spec(("airframe", "total_airframe_hours"))
    assert group.description.startswith("a duration as printed on a sheet")
    assert "hours times 60" not in group.description  # the leaves carry the rest
    assert plan.spec(("airframe", "cycles")).description == "landing cycles"


def test_a_field_description_wins_over_the_models_docstring() -> None:
    class Holder(BaseModel):
        hours: Optional[HhMm] = Field(None, description="airframe hours")

    spec = compile_schema(Holder).spec(("hours",))
    assert spec.description == "airframe hours"
    # ... but its leaves still learn what the model says about them
    assert (
        "hours times 60"
        in compile_schema(Holder).spec(("hours", "total_minutes")).instructions
    )


def test_the_default_basemodel_docstring_is_never_sent() -> None:
    class Bare(BaseModel):
        value: Optional[int] = None

    class Outer(BaseModel):
        bare: Optional[Bare] = None

    plan = compile_schema(Outer)
    assert Bare.__doc__ is None or "Usage" not in str(
        plan.spec(("bare", "value")).instructions
    )
    assert plan.spec(("bare",)).description == "bare"
    assert plan.spec(("bare", "value")).context == "Outer → bare"


def test_a_docstring_that_only_repeats_the_names_adds_nothing() -> None:
    class Address(BaseModel):
        """An address."""

        city: Optional[str] = None

    class Person(BaseModel):
        address: Optional[Address] = None

    plan = compile_schema(Person)
    assert plan.spec(("address", "city")).context == "Person → address"
    assert plan.spec(("address",)).description == "address"


def test_the_items_of_a_list_of_models_describe_the_field() -> None:
    class Gear(BaseModel):
        """One landing gear leg: nose or a main leg, with its overhaul history."""

        position: Optional[str] = None

    class Plane2(BaseModel):
        gear: list[Gear] = Field(default_factory=list)

    plan = compile_schema(Plane2)
    assert plan.spec(("gear",)).description.startswith("one landing gear leg")


def test_a_long_docstring_is_cut_in_the_context_and_the_text_stays_in_the_limit() -> (
    None
):
    Wordy = create_model(
        "Wordy",
        __doc__="Sentence one explains the first thing in some detail. "
        + "Sentence two keeps going and going. " * 30,
        leaf=(Optional[int], None),
    )

    class Root(BaseModel):
        wordy: Optional[Wordy] = None

    spec = compile_schema(Root).spec(("wordy", "leaf"))
    assert len(spec.instructions) <= Limits().max_instruction_chars
    assert spec.instructions.endswith(": leaf")
    assert "Sentence one explains" in spec.context
    assert len(spec.context) < 260
