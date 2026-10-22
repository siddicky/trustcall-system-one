"""Candidate generation, structure recovery and date parts of the TypeSafe stand-in.

Pure code: nothing here calls a model. The candidates are what the model will be asked
to pick among, so the tests pin what is offered per field type (with tables), that a
span is an exact offset into the input, and the cookbooks' arithmetic (the join
thresholds, the date resolution and its review flag).
"""

from datetime import date
from typing import Any

import pytest

from trustcall_system_one._candidates import (
    MAX_OPTIONS,
    Candidate,
    context_snippet,
    pattern_candidates,
)
from trustcall_system_one._dateparts import (
    INFERRED_CONFIDENCE,
    REVIEW_BELOW,
    Part,
    assemble_date,
    assemble_datetime,
    assemble_duration,
    assemble_time,
    date_questions,
    duration_questions,
    time_questions,
)
from trustcall_system_one._structure import (
    build_structure,
    ends_terminal,
    join_questions,
    merge_lines,
    tag_lines,
    to_lines,
)
from trustcall_system_one._wire import ClairicField

NOTE = (
    "From: Dana Whit <dana.whit@acme-corp.com>\n"
    "Reply-To: dana.personal@gmail.com\n"
    "Call (415) 555-0177 or +1 415 555 0142. Site https://acme-corp.com/pay.\n"
    "Total due: $1,315.50 incl 15% tax, 4 x 250 + 80. Invoice INV-2291 for "
    'Emma Thompson\'s "blue mugs": twenty-one items and a dozen eggs, due '
    "August 14, 2027 or next Thursday at 2:30 pm."
)


def field(type_: str = "text", **kwargs: Any) -> ClairicField:
    return ClairicField(
        type=type_, instructions=kwargs.pop("instructions", "x"), **kwargs
    )


def found(type_: str, text: str = NOTE, **kwargs: Any) -> dict[str, Candidate]:
    return {c.text: c for c in pattern_candidates(field(type_, **kwargs), text)}


# --------------------------------------------------------------------------- #
# Spans are exact offsets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type_", ["text", "number", "date"])
def test_every_pattern_candidate_is_an_exact_slice_of_the_input(type_: str) -> None:
    candidates = pattern_candidates(field(type_), NOTE)
    assert candidates
    for c in candidates:
        assert NOTE[c.start : c.end] == c.text
        assert c.occurrences[0] == (c.start, c.end)
        assert f"«{c.text}»" in c.contexts[0]
    # in document order, one per text
    assert [c.start for c in candidates] == sorted(c.start for c in candidates)
    assert len({c.text for c in candidates}) == len(candidates)


def test_a_repeated_text_is_one_candidate_with_several_places() -> None:
    text = "Ann called. Later Ann wrote. Then Ann left."
    ann = found("text", text)["Ann"]
    assert ann.span == (0, 3)
    assert len(ann.occurrences) == 3
    assert len(ann.contexts) == 3


def test_context_marks_the_candidate_and_keeps_line_breaks() -> None:
    snippet = context_snippet(NOTE, 52, 75, 20)
    assert "«dana.personal@gmail.com»" in snippet
    assert "Reply-To" in snippet
    assert context_snippet("abc", 0, 3) == "«abc»"


# --------------------------------------------------------------------------- #
# Text: pattern-shaped values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("dana.whit@acme-corp.com", "email"),
        ("dana.personal@gmail.com", "email"),
        ("https://acme-corp.com/pay", "url"),
        ("(415) 555-0177", "phone"),
        ("+1 415 555 0142", "phone"),
        ("INV-2291", "identifier"),
        ("blue mugs", "quoted"),
        ("Emma Thompson", "name"),
        ("Thompson", "name"),
        ("Dana Whit", "name"),
        ("$1,315.50", "amount"),
        ("15%", "number"),
        ("August 14, 2027", "date"),
        ("next Thursday", "date"),
        ("2:30 pm", "time"),
    ],
)
def test_text_candidates(text: str, kind: str) -> None:
    candidates = found("text")
    assert text in candidates, sorted(candidates)
    assert candidates[text].kind == kind


def test_a_labelled_value_is_its_value() -> None:
    candidates = found("text", "Name: Dana Whit\nCity: Berlin, Germany\n")
    assert candidates["Dana Whit"].kind in ("labelled", "name")
    assert "Berlin" in candidates


def test_digits_inside_a_phone_or_a_date_are_not_separate_candidates() -> None:
    candidates = found("text")
    assert "415" not in candidates
    assert "555-0177" not in candidates
    assert "2027" not in candidates  # inside "August 14, 2027"


def test_a_phrase_after_a_linking_word_is_offered() -> None:
    text = "Tom Becker joins the same week as finance lead."
    lines = to_lines(text)
    structure = build_structure(text, lines, {})
    pieces = {c.text: c for c in structure.candidates}
    assert "finance lead" in pieces
    assert "the same week" in pieces
    assert "same week" in pieces
    assert pieces["finance lead"].id == "S000.p3" or pieces[
        "finance lead"
    ].id.startswith("S000.")
    assert (
        text[pieces["finance lead"].start : pieces["finance lead"].end]
        == "finance lead"
    )


def test_title_with_of_is_one_name() -> None:
    assert "Head of Ops" in found("text", "She started as Head of Ops at Northwind.")


def test_clauses_and_noun_phrases_come_from_the_structure_not_from_patterns() -> None:
    assert "finance lead" not in found("text", "He joins as finance lead.")


def test_stop_words_are_not_names() -> None:
    candidates = found("text", "Thanks, the team met.")
    assert "Thanks" not in candidates
    assert "The" not in candidates


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("$1,315.50", 1315.5),
        ("15%", 15),
        ("4", 4),
        ("250", 250),
        ("twenty-one", 21),
        ("a dozen", 12),
        ("two", 2),
    ],
)
def test_number_candidates(text: str, value: float) -> None:
    note = NOTE + " Two of them."
    text = text if text != "two" else "Two"
    candidates = found("number", note)
    assert candidates[text].value == value


def test_reading_shows_currency_and_percent() -> None:
    candidates = found("number")
    assert candidates["$1,315.50"].reading == "1315.5 USD"
    assert candidates["15%"].reading == "15 percent"


def test_digits_of_ids_dates_and_phones_are_not_numbers() -> None:
    candidates = found("number")
    for text in ("2291", "415", "555", "2027", "14", "2", "30"):
        assert text not in candidates, text


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("5k", 5000),
        ("2.5 million", 2_500_000),
        ("€1.315,50", 1315.5),
        ("-12", -12),
        ("1,000", 1000),
    ],
)
def test_number_forms(text: str, value: float) -> None:
    candidates = found("number", f"It was {text} in all.")
    assert candidates[text].value == value


def test_integer_and_decimals_constrain_the_values() -> None:
    assert "12.5" not in found("number", "It is 12.5 here", integer=True)
    assert found("number", "It is 12.567 here", decimals=1)["12.567"].value == 12.6


def test_an_estimate_stays_in_its_bounds() -> None:
    candidates = found("estimate", "Between 5 and 500 units.", min=0, max=100)
    assert "5" in candidates
    assert "500" not in candidates


def test_a_sum_is_offered_only_when_the_field_asks_for_a_total() -> None:
    text = "4 pallets at $250.00 each, shipping $80.00."
    assert "4 pallets at $250.00 each, shipping $80.00" not in found("number", text)
    asked = found("number", text, instructions="The total amount due")
    assert asked["4 pallets at $250.00 each, shipping $80.00"].value == 1080
    chained = found("number", "That is 4 x 250 + 80 in all.", instructions="the total")
    assert chained["4 x 250 + 80"].value == 1080
    assert chained["4 x 250 + 80"].id == "1080 (the sum of 1000 + 80)"
    assert asked["4 pallets at $250.00 each, shipping $80.00"].id == (
        "1080 (the sum of 1000 + 80)"
    )


def test_a_list_of_numbers_has_no_sums() -> None:
    candidates = found("list", "4 x 250 + 80", items="number", instructions="total")
    assert all(c.kind != "sum" for c in candidates.values())


# --------------------------------------------------------------------------- #
# Dates are found, not read
# --------------------------------------------------------------------------- #


def test_a_date_field_gets_mentions_whose_value_is_their_text() -> None:
    candidates = found("date")
    assert set(candidates) >= {"August 14, 2027", "next Thursday"}
    for c in candidates.values():
        assert c.value == c.text  # nothing is resolved here


@pytest.mark.parametrize(
    "text",
    [
        "2026-08-14",
        "8/14/2026",
        "14.08.2026",
        "14th of August",
        "tomorrow",
        "in 3 days",
    ],
)
def test_date_mentions(text: str) -> None:
    assert text in found("date", f"Due {text}, please.")


def test_a_lowercase_weekday_abbreviation_is_a_word() -> None:
    assert "sat" not in found("date", "He sat down.")


# --------------------------------------------------------------------------- #
# Cap
# --------------------------------------------------------------------------- #


def test_option_limit_is_a_choice_minus_the_none_option() -> None:
    assert MAX_OPTIONS == 254


# --------------------------------------------------------------------------- #
# Structure recovery (cookbook)
# --------------------------------------------------------------------------- #

MEMO = (
    "Migration to the new build system\n"
    "\n"
    "Hi everyone, quick heads up about the build system migration that is\n"
    "happening next week. We have been running the new pipeline in shadow\n"
    "mode for three weeks and the results look solid, so it is time to\n"
    "make the switch for real.\n"
    "\n"
    "The cutover touches three teams, so check whether you are on this\n"
    "list before you plan anything for Monday:\n"
    "The platform team\n"
    "The web client team\n"
)


def test_lines_have_ids_exact_offsets_and_gaps() -> None:
    lines = to_lines(MEMO)
    assert [line.id for line in lines[:3]] == ["L000", "L001", "L002"]
    assert [line.gap for line in lines[:3]] == [False, True, False]
    for line in lines:
        assert MEMO[line.start : line.end] == line.text


def test_the_stitch_state_tags_lines_and_blank_lines() -> None:
    tagged = tag_lines(to_lines(MEMO)).splitlines()
    assert tagged[0] == "L000| Migration to the new build system"
    assert tagged[1] == ""
    assert tagged[2].startswith("L001| Hi everyone")


def test_join_questions_skip_blank_line_pairs_and_use_the_cookbook_wording() -> None:
    lines = to_lines(MEMO)
    questions = join_questions(lines)
    assert "L001" not in questions  # a blank line comes before it
    assert "L002" in questions
    q = questions["L002"]
    assert q["type"] == "noul"
    assert q["instructions"] == (
        "Does line L002 pick up mid-sentence, continuing a sentence left "
        "unfinished at the end of line L001?"
    )
    assert q["criteria"]["false"].startswith("The line begins a new sentence")  # type: ignore[index]


def test_ends_terminal() -> None:
    assert ends_terminal("It ends here.")
    assert ends_terminal('He said "no."')
    assert ends_terminal("Teams:")
    assert not ends_terminal("so it is time to")


def test_two_thresholds_depend_on_how_the_previous_line_ends() -> None:
    lines = to_lines(
        "The cutover touches three teams, so check:\nThe platform team\n"
        "make the\nswitch"
    )
    # L001 follows a colon: 0.22 is below the 0.5 bar. L003 follows a dangling line:
    # 0.39 clears the 0.2 bar
    joins = {"L001": 0.22, "L002": 0.05, "L003": 0.39}
    groups = merge_lines(lines, joins)
    assert [[line.id for line in g] for g in groups] == [
        ["L000"],
        ["L001"],
        ["L002", "L003"],
    ]
    # with one cautious cutoff of 0.5 the dangling line would not have been healed
    cautious = merge_lines(lines, joins, after_dangling=0.5)
    assert [[line.id for line in g] for g in cautious] == [
        ["L000"],
        ["L001"],
        ["L002"],
        ["L003"],
    ]


def test_blocks_sentences_and_pieces_map_to_exact_offsets() -> None:
    lines = to_lines(MEMO)
    joins = {line.id: 0.0 for line in lines}
    joins.update({"L002": 0.77, "L003": 0.62, "L004": 0.39, "L006": 0.5})
    structure = build_structure(MEMO, lines, joins)
    ids = [b.id for b in structure.blocks]
    assert ids[:2] == ["B000", "B001"]
    block = structure.blocks[1]
    assert len(block.lines) == 4  # the hard-wrapped paragraph is one block
    candidate = structure.by_id["B001"]
    assert candidate.kind == "block"
    assert MEMO[candidate.start : candidate.end].startswith("Hi everyone")
    assert MEMO[candidate.start : candidate.end].endswith("switch for real.")
    sentence = next(
        c for c in structure.candidates if c.kind == "sentence" and "shadow" in c.text
    )
    assert "\n" not in sentence.text  # the joined text, whitespace normalised
    assert " ".join(MEMO[sentence.start : sentence.end].split()) == sentence.text
    # a sentence spanning a hard wrap still maps to the original start and end
    assert MEMO[sentence.start : sentence.start + 7] == "We have"
    assert MEMO[sentence.end - 1] == "."
    # tagged state: block ids and sentence ids, one copy of the text
    tagged = structure.tagged()
    assert tagged.startswith("B000| [S000] Migration to the new build system")
    assert "B001| [S001] Hi everyone" in tagged
    assert structure.stitched == len(lines) - len(structure.blocks)


def test_candidate_descriptions_are_short() -> None:
    lines = to_lines(MEMO)
    structure = build_structure(MEMO, lines, {})
    for c in structure.candidates:
        assert c.description is not None
        assert len(c.description) < 140
        assert c.id


# --------------------------------------------------------------------------- #
# Date parts (cookbook): reference date 2026-07-30, a Thursday
# --------------------------------------------------------------------------- #

TODAY = date(2026, 7, 30)


def parts(**choices: Any) -> dict[str, Part]:
    """Parts from ``name=choice`` or ``name=(choice, confidence)``."""
    out = {}
    for name, given in choices.items():
        choice, conf = given if isinstance(given, tuple) else (given, 0.95)
        out[name] = Part(choice, conf, {choice: conf, "none": 1 - conf})
    return out


@pytest.mark.parametrize(
    ("given", "value"),
    [
        (
            {"mode": "absolute", "month": "January", "day": "1", "year": "2025"},
            "2025-01-01",
        ),
        (
            {"mode": "absolute", "month": "August", "day": "14", "year": "none"},
            "2026-08-14",
        ),
        (
            {"mode": "absolute", "month": "January", "day": "5", "year": "none"},
            "2027-01-05",
        ),
        ({"mode": "relative", "day_anchor": "today"}, "2026-07-30"),
        ({"mode": "relative", "day_anchor": "tomorrow"}, "2026-07-31"),
        ({"mode": "relative", "day_anchor": "day_after"}, "2026-08-01"),
        (
            {
                "mode": "relative",
                "day_anchor": "weekday",
                "weekday": "Thursday",
                "week_offset": "next",
            },
            "2026-08-06",
        ),
        (
            {
                "mode": "relative",
                "day_anchor": "weekday",
                "weekday": "Monday",
                "week_offset": "current",
            },
            "2026-07-27",
        ),
        (
            {
                "mode": "relative",
                "day_anchor": "weekday",
                "weekday": "Thursday",
                "week_offset": "none",
            },
            "2026-07-30",
        ),
        (
            {
                "mode": "relative",
                "day_anchor": "weekday",
                "weekday": "Monday",
                "week_offset": "none",
            },
            "2026-08-03",
        ),
    ],
)
def test_the_date_is_resolved_in_code(given: dict[str, str], value: str) -> None:
    result = assemble_date(parts(**given), TODAY)
    assert result.value == value
    # a year or a week the text did not state is inferred, and says so
    inferred = given.get("year") == "none" or given.get("week_offset") == "none"
    assert result.needs_review is inferred
    assert bool(result.inferred) is inferred
    assert result.confidence == (INFERRED_CONFIDENCE if inferred else 0.95)


@pytest.mark.parametrize(
    ("given", "value"),
    [
        ({"mode": "relative", "day_anchor": "yesterday"}, "2026-07-29"),
        (
            {
                "mode": "relative",
                "day_anchor": "weekday",
                "weekday": "Thursday",
                "week_offset": "previous",
            },
            "2026-07-23",
        ),
        (
            {
                "mode": "relative",
                "day_anchor": "weekday",
                "weekday": "Monday",
                "week_offset": "previous",
            },
            "2026-07-20",
        ),
    ],
)
def test_past_dates_are_stated_not_inferred(given: dict[str, str], value: str) -> None:
    result = assemble_date(parts(**given), TODAY)
    assert result.value == value
    assert result.inferred == ()
    assert result.needs_review is False
    assert result.confidence == 0.95


def test_the_options_offer_the_past() -> None:
    questions = date_questions("the date")
    assert "yesterday" in questions["day_anchor"]["criteria"]
    assert "previous" in questions["week_offset"]["criteria"]


def test_a_year_less_date_is_flagged_and_kept_under_the_reask_bar() -> None:
    result = assemble_date(
        parts(mode="absolute", month="August", day="14", year="none"), TODAY
    )
    assert result.inferred == ("year",)
    assert "inferred year" in result.note
    assert result.needs_review is True
    assert result.confidence is not None
    assert result.confidence < 0.85  # the library's low-confidence bar: re-asked


@pytest.mark.parametrize(
    ("given", "note"),
    [
        ({"mode": "none"}, "no such date stated"),
        (
            {"mode": "absolute", "month": "none", "day": "none", "year": "none"},
            "incomplete",
        ),
        (
            {"mode": "absolute", "month": "February", "day": "30", "year": "2026"},
            "impossible",
        ),
        (
            {"mode": "absolute", "month": "May", "day": "3", "year": "out_of_range"},
            "year outside",
        ),
        ({"mode": "relative", "day_anchor": "none"}, "not read"),
    ],
)
def test_no_date_when_the_parts_do_not_add_up(given: dict[str, str], note: str) -> None:
    result = assemble_date(parts(**given), TODAY)
    assert result.value is None
    assert result.needs_review is True
    assert note in result.note
    assert result.p_null > 0.0


def test_confidence_is_the_weakest_part_used_and_flags_review() -> None:
    given = parts(
        mode="absolute",
        month=("March", 0.9),
        day=("3", 0.55),
        year=("2026", 0.97),
        weekday=("Monday", 0.01),  # not used for an absolute date
    )
    result = assemble_date(given, TODAY)
    assert result.value == "2026-03-03"
    assert result.confidence == pytest.approx(0.55)
    assert result.needs_review is (REVIEW_BELOW > 0.55)
    assert assemble_date(given, TODAY, review_below=0.5).needs_review is False


def test_null_probability_comes_from_the_mode_answer() -> None:
    mode = Part("none", 0.9, {"absolute": 0.04, "relative": 0.06, "none": 0.9})
    assert assemble_date({"mode": mode}, TODAY).p_null == pytest.approx(0.9)
    sure = Part("absolute", 0.9, {"absolute": 0.95, "relative": 0.02, "none": 0.03})
    given = {**parts(month="May", day="5", year="2026"), "mode": sure}
    assert assemble_date(given, TODAY).p_null == pytest.approx(0.03)


def test_time_parts() -> None:
    result = assemble_time(parts(mode="stated", hour="14", minute="30"))
    assert result.value == "14:30:00"
    assert (
        assemble_time(parts(mode="stated", hour="12", minute="none")).value
        == "12:00:00"
    )
    assert assemble_time(parts(mode="none")).value is None
    assert assemble_time(parts(mode="stated", hour="none", minute="none")).value is None


def test_datetime_needs_both_parts() -> None:
    day = parts(mode="absolute", month="August", day="14", year="2027")
    clock = parts(mode="stated", hour="14", minute="30")
    assert assemble_datetime(day, clock, TODAY).value == "2027-08-14T14:30:00"
    assert assemble_datetime(day, parts(mode="none"), TODAY).value is None
    assert assemble_datetime(parts(mode="none"), clock, TODAY).value is None


def test_duration_parts_sum_in_the_fields_unit() -> None:
    given = parts(mode="stated", hours="2", minutes="30", days="none", seconds="none")
    assert (
        assemble_duration(given, "minutes", integer=False, decimals=None).value == 150
    )
    assert assemble_duration(given, "hours", integer=False, decimals=None).value == 2.5
    assert assemble_duration(given, "hours", integer=True, decimals=None).value is None
    assert (
        assemble_duration(given, "seconds", integer=True, decimals=None).value == 9000
    )
    half = parts(mode="stated", hours="0.5")
    assert assemble_duration(half, "minutes", integer=True, decimals=None).value == 30
    big = parts(mode="stated", days="out_of_range")
    assert assemble_duration(big, "days", integer=False, decimals=None).value is None
    assert (
        assemble_duration(
            parts(mode="none"), "days", integer=False, decimals=None
        ).value
        is None
    )


def test_the_part_questions_are_the_cookbooks() -> None:
    questions = date_questions("the deadline")
    assert list(questions) == [
        "mode",
        "month",
        "day",
        "year",
        "day_anchor",
        "weekday",
        "week_offset",
    ]
    assert questions["year"]["criteria"]["1900"] is None
    assert questions["year"]["criteria"]["2050"] is None
    assert "out_of_range" in questions["year"]["criteria"]
    assert len(questions["year"]["criteria"]) == 153
    assert "How is the deadline written?" in questions["mode"]["instructions"]
    assert set(time_questions("x")) == {"mode", "hour", "minute"}
    assert list(duration_questions("x"))[:3] == ["mode", "years", "months"]
    for group in (date_questions("x"), time_questions("x"), duration_questions("x")):
        for q in group.values():
            assert q["type"] == "choice"
            assert 2 <= len(q["criteria"]) <= 255


def _mode(choice: str, none: float) -> Part:
    return Part(choice, 1 - none, {choice: 1 - none, "none": none})


def test_a_datetime_is_null_when_either_part_is_probably_not_stated() -> None:
    day = {
        **parts(month="August", day="14", year="2027"),
        "mode": _mode("absolute", 0.02),
    }
    unstated = {**parts(hour="14", minute="30"), "mode": _mode("stated", 0.9)}
    stated = {**parts(hour="14", minute="30"), "mode": _mode("stated", 0.03)}
    # a date, and a time that is probably not in the text (the time part says so)
    missed = assemble_datetime(day, unstated, TODAY)
    assert missed.p_null == pytest.approx(0.9)
    # the datetime is stated: the larger of the two small chances
    sure = assemble_datetime(day, stated, TODAY)
    assert sure.value == "2027-08-14T14:30:00"
    assert sure.p_null == pytest.approx(0.03)
    # ... and the date part is the doubtful one
    doubtful = {**day, "mode": _mode("absolute", 0.7)}
    assert assemble_datetime(doubtful, stated, TODAY).p_null == pytest.approx(0.7)


def test_a_datetime_without_a_value_is_null_by_its_least_sure_part() -> None:
    day = {**parts(month="August", day="14", year="2027"), "mode": _mode("none", 0.8)}
    clock = {**parts(hour="9", minute="0"), "mode": _mode("stated", 0.05)}
    result = assemble_datetime(day, clock, TODAY)
    assert result.value is None
    assert result.p_null >= 0.8


def test_the_duration_counts_are_complete_to_120() -> None:
    counts = duration_questions("x")["hours"]["criteria"]
    for n in range(1, 121):  # every whole count, 61 to 89 included
        assert str(n) in counts
    assert "out_of_range" in counts


# --------------------------------------------------------------------------- #
# Review fixes: line breaks, instructions that are not names, brackets
# --------------------------------------------------------------------------- #


def test_a_phone_number_does_not_run_across_a_line_break() -> None:
    text = "Order 4155550100\n2024 was the year.\nCall 415 555 0100 now."
    phones = [
        c.text for c in pattern_candidates(field("text"), text) if c.kind == "phone"
    ]
    assert "415 555 0100" in phones
    assert not any("\n" in p for p in phones)
    assert not any("2024" in p for p in phones)


@pytest.mark.parametrize(
    ("text", "joined"),
    [
        ("Due August\n14 for all", "August\n14"),
        ("Due on 14\nAugust 2027 sharp", "14\nAugust"),
        ("Meet next\nThursday", "next\nThursday"),
        ("It is 2:30\npm", "2:30\npm"),
    ],
)
def test_a_date_or_time_does_not_run_across_a_line_break(
    text: str, joined: str
) -> None:
    for kind in ("date", "text"):
        assert not any("\n" in c.text for c in pattern_candidates(field(kind), text))
    assert joined not in {c.text for c in pattern_candidates(field("date"), text)}


def test_a_date_on_one_line_is_still_found() -> None:
    assert "August 14" in found("date", "Due August 14 for all")
    assert "2:30 pm" in found("text", "It is 2:30 pm")


@pytest.mark.parametrize(
    "word", ["Call", "Please", "Contact", "Reply", "Send", "Thanks", "Total"]
)
def test_the_word_that_opens_an_instruction_is_not_a_name(word: str) -> None:
    candidates = found("text", f"{word} Dana Whit about it.")
    assert word not in candidates
    assert "Dana Whit" in candidates  # the person after the verb is still a name
    assert f"{word} Dana Whit" not in candidates


def test_clauses_never_have_unbalanced_brackets() -> None:
    text = (
        "The fee (see the table, section 4) is due, and (as agreed) paid [late, again]."
    )
    structure = build_structure(text, to_lines(text), {})
    for c in structure.candidates:
        assert c.text.count("(") == c.text.count(")"), c.text
        assert c.text.count("[") == c.text.count("]"), c.text
