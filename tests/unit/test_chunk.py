"""Tests for chunk splitting and the per-field merge."""

import itertools

import pytest

from trustcall_system_one._chunk import (
    DEFAULT_CHUNK_CHARS,
    Part,
    agree,
    chunk_size,
    identity_state,
    merge_field,
    merge_parts,
    part_of,
    split_ranges,
)
from trustcall_system_one._render import render_clairic
from trustcall_system_one._wire import ClairicAnswer


def answer(value: object, **extra: object) -> ClairicAnswer:
    return ClairicAnswer(type="text", value=value, **extra)


def part(
    value: object,
    start: int = 0,
    *,
    span: list[int] | None = None,
    spans: list[list[int]] | None = None,
    **extra: object,
) -> Part:
    """A part whose chunk starts at ``start`` (spans given in original offsets)."""
    raw = answer(value, span=span, spans=spans, **extra)
    return part_of(raw, identity_state("x" * 10_000), start)


# --------------------------------------------------------------------------- #
# split_ranges
# --------------------------------------------------------------------------- #


def test_a_range_no_longer_than_a_chunk_is_one_chunk() -> None:
    assert split_ranges([(0, 100)], 100, 10) == [(0, 100)]
    assert split_ranges([(5, 50)], 100, 10) == [(5, 50)]


def test_chunks_are_consecutive_and_overlap() -> None:
    chunks = split_ranges([(0, 1000)], 300, 50)
    assert chunks[0] == (0, 300)
    assert chunks[-1][1] == 1000
    for (_, end), (start, _) in itertools.pairwise(chunks):
        assert end - start == 50
    assert all(end - start <= 300 for start, end in chunks)
    covered: set[int] = set()
    for start, end in chunks:
        covered.update(range(start, end))
    assert covered == set(range(1000))


def test_every_range_is_split_on_its_own() -> None:
    chunks = split_ranges([(0, 250), (1000, 1100)], 100, 20)
    assert chunks[-1] == (1000, 1100)
    assert all(not (a < 1000 < b) for a, b in chunks)
    assert [c for c in chunks if c[0] < 1000][-1][1] == 250


@pytest.mark.parametrize(
    ("size", "overlap"), [(3, 3), (3, 4), (0, 0), (-5, 0), (10, -1)]
)
def test_an_overlap_that_stops_progress_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk"):
        split_ranges([(0, 10)], size, overlap)


def test_the_largest_valid_overlap_still_makes_progress() -> None:
    chunks = split_ranges([(0, 10)], 3, 2)
    assert chunks[-1][1] == 10 and len(chunks) == 8


# --------------------------------------------------------------------------- #
# chunk_size
# --------------------------------------------------------------------------- #


def test_chunk_size_comes_from_what_was_read() -> None:
    assert chunk_size(5000, 26_000, 100) == 4000
    assert chunk_size(None, 26_000, 100) == DEFAULT_CHUNK_CHARS


def test_chunk_size_always_splits_and_never_drops_below_the_floor() -> None:
    assert chunk_size(None, 1000, 20) == 600  # below the range: at least two chunks
    assert chunk_size(10, 1000, 100) == 201  # 2 * overlap + 1
    assert chunk_size(50_000, 26_000, 100) == int(26_000 * 0.6)


# --------------------------------------------------------------------------- #
# part_of
# --------------------------------------------------------------------------- #


def test_part_of_maps_spans_through_the_chunk_state() -> None:
    text = "0123456789" * 30
    rendered = render_clairic(text, label="x", windows=[(100, 200)])
    offset = rendered.segments[0][0]
    raw = answer("890", span=[offset + 8, offset + 11])
    mapped = part_of(raw, rendered, 100)
    assert mapped.span == (108, 111)
    assert text[108:111] == "890"
    listed = answer(["a", "b"], spans=[[offset, offset + 2], [0, 3]])  # 2nd: in focus
    assert part_of(listed, rendered, 100).spans == ((100, 102), None)
    assert part_of(answer("v"), rendered, 100).span is None
    assert part_of(answer("v"), None, 100).span is None


# --------------------------------------------------------------------------- #
# merge_parts
# --------------------------------------------------------------------------- #


def test_nothing_merges_to_nothing() -> None:
    assert merge_parts([]) is None


def test_all_null_reports_the_lowest_p_null() -> None:
    # the lowest, so a borderline null (the 0.25-0.75 band) can still be checked
    merged = merge_parts([part(None, p_null=0.99), part(None, p_null=0.45)])
    assert merged is not None and merged.value is None and merged.p_null == 0.45
    assert merged.span is None and merged.spans is None


def test_all_null_prefers_an_answer_that_states_its_p_null() -> None:
    bare = part(None, 0)
    bare.answer.p_null = None
    merged = merge_parts([bare, part(None, 10, p_null=0.8)])
    assert merged is not None and merged.p_null == 0.8


def test_non_null_beats_null() -> None:
    merged = merge_parts(
        [part(None, p_null=0.99), part("Paris", span=[5, 10], confidence=0.6)]
    )
    assert merged is not None
    assert merged.value == "Paris" and merged.span == [5, 10]


def test_higher_confidence_wins_and_ties_go_to_the_earlier_chunk() -> None:
    low = part("low", span=[1, 4], confidence=0.6)
    high = part("high", 1000, span=[1001, 1005], confidence=0.9)
    merged = merge_parts([low, high])
    assert merged is not None and merged.value == "high" and merged.span == [1001, 1005]
    tie_a = part("a", span=[1, 2], confidence=0.8)
    tie_b = part("b", 1000, span=[1001, 1002], confidence=0.8)
    first = merge_parts([tie_a, tie_b])
    assert first is not None and first.value == "a"


def test_every_scalar_candidate_is_kept_ranked_by_confidence() -> None:
    first = part("London", span=[1, 7], confidence=0.97)
    second = part("Paris", 1000, span=[1001, 1006], confidence=0.9)
    third = part("Rome", 2000, span=[2001, 2005], confidence=0.93)
    merged = merge_field([first, second, third])
    assert merged.conflict
    assert [c.answer.value for c in merged.candidates] == ["London", "Rome", "Paris"]
    assert merged.answer is not None and merged.answer.value == "London"
    assert merged.answer.span == [1, 7]


def test_agreeing_candidates_are_one_value() -> None:
    merged = merge_field(
        [
            part("Ada  Lovelace", span=[1, 14], confidence=0.8),
            part("ada lovelace", 1000, span=[1001, 1013], confidence=0.95),
            part("Ada Lovelaces", 2000, span=[2001, 2014], confidence=0.7),
        ]
    )
    assert not merged.conflict
    assert merged.answer is not None and merged.answer.value == "ada lovelace"
    assert merged.answer.span == [1001, 1013]


def test_numbers_agree_by_value() -> None:
    merged = merge_field([part(3, confidence=0.8), part(3.0, 1000, confidence=0.7)])
    assert not merged.conflict
    assert merge_field([part(3), part(4, 1000)]).conflict


def test_a_single_candidate_and_null_chunks_never_conflict() -> None:
    merged = merge_field([part(None, p_null=0.9), part("x", 10, confidence=0.5)])
    assert not merged.conflict and merged.candidates == merged.candidates[:1]
    assert merged.answer is not None and merged.answer.value == "x"
    assert merge_field([]).answer is None and not merge_field([]).conflict


def test_agree_helper_is_shared() -> None:
    from trustcall_system_one._verify import agree as verify_agree

    assert verify_agree is agree
    assert agree("abc", "ABC") and not agree("abc", "xyz")
    assert agree(["a", "b"], ["b", "a"]) and not agree(["a"], ["a", "b"])
    assert agree(None, None) and not agree(None, "a")


def test_scalars_without_a_confidence_still_merge() -> None:
    merged = merge_parts([part("a"), part("b", 1000)])
    assert merged is not None and merged.value == "a"


def test_lists_are_united_in_input_order_with_span_dedup() -> None:
    first = part(["beta", "alpha"], 0, spans=[[500, 504], [10, 15]], confidence=0.9)
    second = part(
        ["beta", "gamma"], 400, spans=[[500, 504], [900, 905]], confidence=0.8
    )
    merged = merge_parts([first, second])
    assert merged is not None
    assert merged.value == ["alpha", "beta", "gamma"]
    assert merged.spans == [[10, 15], [500, 504], [900, 905]]
    assert merged.confidence == 0.8  # the least certain chunk
    assert merged.span is None


def test_overlapping_list_spans_keep_the_longer_one() -> None:
    # chunk 1 saw "Ada Love" cut at the chunk edge, chunk 2 the whole name
    merged = merge_parts(
        [
            part(["Ada Love"], 0, spans=[[100, 108]], confidence=0.99),
            part(["Ada Lovelace"], 90, spans=[[100, 112]], confidence=0.8),
        ]
    )
    assert merged is not None
    assert merged.value == ["Ada Lovelace"] and merged.spans == [[100, 112]]
    reverse = merge_parts(
        [
            part(["Ada Lovelace"], 0, spans=[[100, 112]], confidence=0.8),
            part(["Ada Love"], 90, spans=[[100, 108]], confidence=0.99),
        ]
    )
    assert reverse is not None and reverse.value == ["Ada Lovelace"]


def test_a_contained_span_loses_to_the_container() -> None:
    merged = merge_parts(
        [
            part(["Lovelace"], 0, spans=[[104, 112]]),
            part(["Ada Lovelace"], 90, spans=[[100, 112]]),
            part(["Ada"], 200, spans=[[100, 103]]),
        ]
    )
    assert merged is not None and merged.value == ["Ada Lovelace"]


def test_one_long_span_replaces_every_shorter_one_it_overlaps() -> None:
    merged = merge_parts(
        [
            part(["Ada", "Lovelace"], 0, spans=[[100, 103], [104, 112]]),
            part(["Ada Lovelace"], 90, spans=[[100, 112]]),
        ]
    )
    assert merged is not None
    assert merged.value == ["Ada Lovelace"] and merged.spans == [[100, 112]]


def test_equal_spans_prefer_the_more_confident_chunk() -> None:
    merged = merge_parts(
        [
            part(["red"], 0, spans=[[10, 13]], confidence=0.6),
            part(["Red"], 5, spans=[[10, 13]], confidence=0.9),
        ]
    )
    assert merged is not None and merged.value == ["Red"]


def test_overlapping_spans_are_duplicates_even_if_the_text_differs() -> None:
    merged = merge_parts(
        [
            part(["Ada Lovelace"], 0, spans=[[10, 22]]),
            part(["Ada"], 5, spans=[[10, 13]]),
        ]
    )
    assert merged is not None and merged.value == ["Ada Lovelace"]


def test_the_same_text_at_two_places_is_two_elements() -> None:
    merged = merge_parts(
        [
            part(["red"], 0, spans=[[10, 13]]),
            part(["red"], 900, spans=[[950, 953]]),
        ]
    )
    assert merged is not None and merged.value == ["red", "red"]


def test_lists_without_spans_dedupe_by_text_and_follow_chunk_order() -> None:
    merged = merge_parts([part(["B", "a"], 0), part(["b", "c"], 500), part(["z"], 100)])
    assert merged is not None
    assert merged.value == ["B", "a", "z", "c"]  # ordered by chunk start
    assert merged.spans is None


def test_a_null_chunk_does_not_hide_the_others_list() -> None:
    merged = merge_parts([part(None, p_null=0.99), part(["x"], 10, spans=[[12, 13]])])
    assert merged is not None and merged.value == ["x"]
