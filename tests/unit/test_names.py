"""Tests for wire-name allocation."""

import hashlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trustcall_system_one._names import (
    CLAIRIC_FIELD_RE,
    CLEF_ID_RE,
    FieldNameMap,
    FieldNameMapBuilder,
    Path,
    pretty,
    wire_name,
)

LONG_PATH: Path = (
    "organization",
    "departments",
    0,
    "team_members",
    "primary_contact_information",
    "mobile_phone_number_with_country_code",
)


def test_pretty_joins_segments_with_dots() -> None:
    assert pretty(("author", "name")) == "author.name"
    assert pretty(("tags", 0, "label")) == "tags.0.label"
    assert pretty(()) == ""


def test_regexes_match_the_published_patterns() -> None:
    assert CLAIRIC_FIELD_RE.pattern == r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$"
    assert CLEF_ID_RE.pattern == r"^[A-Za-z0-9_.\-]{1,100}$"
    assert CLAIRIC_FIELD_RE.match("a" * 64)
    assert not CLAIRIC_FIELD_RE.match("a" * 65)
    assert not CLAIRIC_FIELD_RE.match("1a")
    assert CLEF_ID_RE.match("1" * 100)
    assert not CLEF_ID_RE.match("1" * 101)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("name",), "name"),
        (("author", "name"), "author.name"),
        (("tags", 0), "tags.0"),
        (("due date",), "due_date"),
        (("a  b",), "a_b"),
        (("a__b",), "a_b"),
        (("__x__",), "x"),
        (("a.b",), "a_b"),
        (("a/b?c",), "a_b_c"),
        (("naïve",), "na_ve"),
        (("-x",), "f_-x"),
        (("",), "_"),
        (("___",), "_"),
        (("a", "", "b"), "a._.b"),
        ((0,), "f_0"),
        (("9lives",), "f_9lives"),
    ],
)
def test_sanitisation_for_clairic(path: Path, expected: str) -> None:
    name = wire_name(path, target="clairic", taken=set())
    assert name == expected
    assert CLAIRIC_FIELD_RE.fullmatch(name)


def test_clef_names_do_not_need_a_letter_first() -> None:
    assert wire_name((0,), target="clef", taken=set()) == "0"
    assert wire_name(("9lives",), target="clef", taken=set()) == "9lives"
    assert wire_name(("-x",), target="clef", taken=set()) == "-x"


def test_empty_path_and_unknown_target_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        wire_name((), target="clairic", taken=set())
    with pytest.raises(ValueError, match="unknown"):
        wire_name(("a",), target="other", taken=set())  # type: ignore[arg-type]


def test_long_path_has_a_pinned_name() -> None:
    # Pinned on purpose: a refactor must not silently change wire names.
    assert (
        wire_name(LONG_PATH, target="clairic", taken=set())
        == "organization_mobile_phone_number_with_7da93d"
    )
    assert (
        wire_name(("a" * 70,), target="clairic", taken=set())
        == "aaaaaaaaaaaa_aaaaaaaaaaaaaaaaaaaaaaaa_e137eb"
    )


def test_digest_is_blake2s_of_the_unit_separated_path() -> None:
    digest = hashlib.blake2s(
        "\x1f".join(str(s) for s in LONG_PATH).encode(), digest_size=3
    ).hexdigest()
    assert digest == "7da93d"
    assert wire_name(LONG_PATH, target="clairic", taken=set()).endswith(f"_{digest}")


def test_clairic_limit_is_tighter_than_clef() -> None:
    path: Path = ("x" * 80,)
    assert wire_name(path, target="clef", taken=set()) == "x" * 80
    short = wire_name(path, target="clairic", taken=set())
    assert len(short) <= 64
    assert short != "x" * 80


def test_taken_name_forces_the_short_form_and_widens_on_collision() -> None:
    path: Path = ("a b",)
    assert wire_name(path, target="clairic", taken=set()) == "a_b"
    short = wire_name(path, target="clairic", taken={"a_b"})
    assert short == "a_b_a_b_9e6d48"
    wide = wire_name(path, target="clairic", taken={"a_b", short})
    assert wide != short
    assert wide.startswith("a_b_a_b_")
    assert len(wide.rsplit("_", 1)[1]) == 8
    wider = wire_name(path, target="clairic", taken={"a_b", short, wide})
    assert len(wider.rsplit("_", 1)[1]) == 12
    widest_taken = {"a_b", short, wide, wider}
    widest = wire_name(path, target="clairic", taken=widest_taken)
    assert len(widest.rsplit("_", 1)[1]) == 16
    with pytest.raises(ValueError, match="free"):
        wire_name(path, target="clairic", taken=widest_taken | {widest})


def test_wire_name_does_not_mutate_taken() -> None:
    taken: set[str] = {"x"}
    wire_name(("y",), target="clairic", taken=taken)
    assert taken == {"x"}


def test_suffix_is_appended_after_a_dot() -> None:
    assert (
        wire_name(("kind",), target="clef", taken=set(), suffix="__type")
        == "kind.__type"
    )


def test_builder_resolves_collisions_and_maps_both_ways() -> None:
    builder = FieldNameMapBuilder("clairic")
    first = builder.add(("a b",))
    second = builder.add(("a_b",))
    assert first == "a_b"
    assert second != first
    names = builder.build()
    assert names.path_to_name == {("a b",): first, ("a_b",): second}
    assert names.name_to_path[second] == ("a_b",)
    assert names.pretty(second) == "a_b"
    assert names.shortened == {second}
    assert names.target == "clairic"
    assert len(names) == 2
    assert ("a b",) in names
    assert first in names


def test_builder_flags_names_shortened_for_length() -> None:
    builder = FieldNameMapBuilder("clairic")
    short = builder.add(("x" * 80,))
    plain = builder.add(("ok",))
    assert builder.shortened == {short}
    assert plain not in builder.build().shortened


def test_builder_rejects_a_repeated_path() -> None:
    builder = FieldNameMapBuilder("clef")
    builder.add(("a",))
    with pytest.raises(ValueError, match="already"):
        builder.add(("a",))


def test_name_map_is_read_only() -> None:
    names = FieldNameMapBuilder("clef").build()
    with pytest.raises(TypeError):
        names.path_to_name[("a",)] = "a"  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate"):
        FieldNameMap("clef", [(("a",), "n", False), (("b",), "n", False)])


_segments = st.one_of(
    st.text(max_size=90),
    st.integers(min_value=0, max_value=10**6),
    st.sampled_from(["name", "id", "a_b", "a b", "x" * 70, "__", "-"]),
)
_paths = st.lists(
    st.tuples(_segments)
    | st.tuples(_segments, _segments)
    | st.lists(_segments, min_size=1, max_size=6).map(tuple),
    min_size=1,
    max_size=25,
    unique=True,
)


@pytest.mark.parametrize("target", ["clairic", "clef"])
@given(paths=_paths)
def test_every_name_matches_its_regex_and_maps_back(
    target: str, paths: list[Path]
) -> None:
    builder = FieldNameMapBuilder(target)  # type: ignore[arg-type]
    names = [builder.add(path) for path in paths]
    mapping = builder.build()
    regex = CLAIRIC_FIELD_RE if target == "clairic" else CLEF_ID_RE
    assert len(set(names)) == len(paths)
    for path, name in zip(paths, names, strict=True):
        assert regex.fullmatch(name), name
        assert mapping.name_to_path[name] == path
        assert mapping.path_to_name[path] == name
        assert mapping.pretty(name) == pretty(path)


@given(path=st.lists(_segments, min_size=1, max_size=5).map(tuple))
def test_names_are_deterministic(path: Path) -> None:
    first = wire_name(path, target="clairic", taken=set())
    assert first == wire_name(path, target="clairic", taken=set())


def test_add_all_gives_every_colliding_path_the_digest_form() -> None:
    entries: list[tuple[Path, str | None]] = [
        (("a b",), None),
        (("a_b",), None),
        (("c",), None),
    ]
    for ordering in (entries, list(reversed(entries))):
        builder = FieldNameMapBuilder("clairic")
        names = dict(
            zip((e[0] for e in ordering), builder.add_all(ordering), strict=True)
        )
        assert names[("c",)] == "c"
        assert names[("a b",)] == "a_b_a_b_9e6d48"
        assert names[("a_b",)].startswith("a_b_a_b_")
        assert names[("a_b",)] != names[("a b",)]
        assert "a_b" not in names.values()
        assert builder.build().shortened == {names[("a b",)], names[("a_b",)]}


def test_add_all_treats_int_and_str_segments_with_the_same_text_as_a_collision() -> (
    None
):
    builder = FieldNameMapBuilder("clef")
    one, two = builder.add_all([(("a", 1), None), (("a", "1"), None)])
    assert one != two
    assert "a.1" not in (one, two)
    assert builder.build().name_to_path[one] == ("a", 1)


def test_add_all_avoids_names_from_earlier_calls() -> None:
    builder = FieldNameMapBuilder("clairic")
    assert builder.add(("x",)) == "x"
    (second,) = builder.add_all([(("x ",), None)])
    assert second != "x"


def test_add_all_rejects_repeats_within_and_across_batches() -> None:
    builder = FieldNameMapBuilder("clairic")
    with pytest.raises(ValueError, match="already"):
        builder.add_all([(("a",), None), (("a",), None)])
    builder.add_all([(("b",), None)])
    with pytest.raises(ValueError, match="already"):
        builder.add_all([(("b",), None)])


def test_add_all_shortens_long_names_and_keeps_the_suffix_form() -> None:
    builder = FieldNameMapBuilder("clef")
    long, typed = builder.add_all([(("x" * 120,), None), (("kind",), "__type")])
    assert len(long) <= 100 and long != "x" * 120
    assert typed == "kind.__type"
