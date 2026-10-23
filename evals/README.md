# Evals

Three suites share one case format (`evals/model.py`), one scorer and one runner.

| suite | file | cases | use |
| --- | --- | --- | --- |
| dev | `evals/cases.py` (`CASES`) | 29 | tuning: sweeps, threshold and prompt changes, choosing a provider combination |
| holdout | `evals/holdout_cases.py` (`HOLDOUT_CASES`) | 30 | spent: it was used for diagnosis, so it no longer measures generalisation cleanly |
| holdout2 | `evals/holdout2_cases.py` (`HOLDOUT2_CASES`) | 30 | **the clean held-out set from now on**: measuring a combination that was chosen on the dev suite |

`--suite default|holdout|holdout2` of `python -m evals` picks the suite. `run`, `list`
and `combo` take any of them; `search` (including `--calibrate`) and `sweep` refuse
both held-out suites, because choosing on them would make them dev sets.

## The holdout2 suite

`HOLDOUT2_CASES` is the clean held-out set. It was written blind: from the case
format, the scorer and `docs/DESIGN.md` alone, with no live result, report or
cassette and none of the first holdout's cases in view, and it has never been run
live. It has its own schemas, texts and
domains, all fictional: insurance claims, school field-trip permission forms,
fleet vehicle service logs, restaurant supplier orders, library book loans, IT
asset inventory, conference speaker bios, soil sample tracking (lab work,
non-medical), boat charter bookings and utility meter readings, three cases each.
A test keeps it disjoint from the dev suite and from the first holdout in case ids,
schema names (nested ones too) and text (no six-word run is shared).

It covers one-insert extraction of every scalar type (Decimal money, absolute,
relative and month-year dates, times, durations, bool, `Literal`, `Enum`, optionals
the text never mentions), a sub-model and nested `list[Model]` (markdown tables,
rows named by codes or numbers), many-entity extraction (short forms, pronouns,
non-entities, similar names), updates (only the mentioned fields change, a
correction, a rename, appended notes, a refused relative edit, a requested delete,
a "no longer" that is not a delete, an update/delete conflict), bleed traps, a
dotted identity (`asset.tag`, for an insert and for matching existing records) and
messy inputs (an email thread, a chat log, OCR line breaks, workshop shorthand, a
`TBD` placeholder that must come out null).

Where the library documents a limit, the gold follows the documented behaviour (and
the case's comment and `note` say so): a month and a year with no day is not
resolved to a date (null), a relative edit ("three more guests") is refused with
`relative_edit_unsupported`, text values are verbatim spans, an input that both
changes and removes a record is a `conflict_update_delete` (neither is applied; the
case says so with `expect_skipped`), and a dotted identity is not asked for by value
(its records are named by the words the text uses for them).

Every case has a scripted `FakeWorld`, so it runs offline like the dev cases; a
live run scores what the real models do on the same texts.

Run it offline (no credentials, no network):

```bash
uv run python -m evals run --suite holdout2 --quiet
uv run python -m evals combo --spec spec.json --suite holdout2 --quiet
uv run python -m evals.holdout2_cases
```

or from Python: `from evals.holdout2_cases import HOLDOUT2_CASES` and pass the list
to `evals.runner.run_suite`.

### Rule: never tune on holdout2, never diagnose with it

holdout2 exists to give an honest number for the combinations that a search over the
dev suite picked, and it stays honest only while nothing is fitted to it, and while
nothing about the library is learnt from its failures. The first holdout stopped
being clean exactly that way. So:

- Never tune on it. Do not sweep a parameter, compare presets, choose a provider or
  a combination, or pick a threshold on holdout2. Choose on the dev suite, then run
  holdout2 once per chosen combination.
- Never use it for diagnosis. Do not read its per-case failures, live traces,
  cassettes or probabilities to find out why something went wrong, and do not change
  the library, a prompt, a threshold or the dev suite in response to a particular
  holdout2 case. A failing case is a measurement, not a bug report. If a real library
  bug is suspected, reproduce it on a new dev case with its own text and fix against
  that.
- Do not edit a case, its gold, a threshold, a prompt or a schema after seeing how a
  provider scores on it. Do not add or drop cases to move a score.
- The only valid change to a case is a fix of its gold when the gold is wrong from
  the text alone (or from the documented behaviour it follows). Make it in its own
  commit that says why, and re-run every combination that was already measured.
- Never record cassettes of holdout2 into a directory that the dev suite or the
  tests replay from; a holdout2 result directory is named `...-holdout2-...` so it
  never lands among the dev ones.
- When the next clean set is needed, write it blind like this one and retire this
  one the same way the first was retired.

## The holdout suite (spent)

The first held-out set. It was used for diagnosis, so it no longer measures
generalisation cleanly: report holdout2 for that. It stays as a regression set.

`HOLDOUT_CASES` is an independent dataset in the same format: new schemas,
new texts and new domains (logistics, healthcare scheduling, real estate listings,
aviation maintenance, SaaS support, a recipe and a nutrition label, event
registrations). All data is fictional. It covers one-insert extraction of every
scalar type, nested `list[Model]` items, many-entity extraction (short forms,
pronouns, non-entities, similar names), updates (renames, corrections, appended
notes, a refused relative edit, a requested delete, a "no longer" that is not a
delete, an update/delete conflict), bleed traps, a dotted identity
(`identification.serial`) and messy inputs (an email thread, a meeting transcript,
an OCR-style record card, a CRM note in shorthand).

Every case has a scripted `FakeWorld`, so it runs offline like the dev cases; a
live run scores what the real models do on the same texts.

Run it offline (no credentials, no network):

```bash
uv run python -m evals.holdout_cases
```

or from Python: `from evals.holdout_cases import HOLDOUT_CASES` and pass the list to
`evals.runner.run_suite`. `--suite holdout` of `python -m evals` (`run`, `list`,
`combo`) selects the same 30 cases through `evals/suites.py`:

```bash
uv run python -m evals run --suite holdout --quiet
uv run python -m evals combo --suite holdout --quiet
```

### Rule: never tune on the holdout

The holdout exists to give an honest number for the combinations that a search
over the dev suite picked. That number is only honest while nothing was fitted to
it. So:

- Do not sweep a parameter, compare presets or choose a provider on the holdout.
  Choose on the dev suite, then run the holdout once per chosen combination.
- Do not edit a case, its gold, a threshold, a prompt or a schema description after
  seeing how a provider scores on it. Do not add or drop cases to move a score.
- Do not read the holdout's live results while changing the library or the dev
  suite. If a holdout failure points at a real library bug, record it as a new dev
  case (with its own text) and fix against that.
- The only valid change to a holdout case is a fix of its gold when the gold is
  wrong from the text alone. Make it in its own commit that says why, and re-run
  every combination that was already measured.
- Never record cassettes of the holdout into a directory that the dev suite or the
  tests replay from.

### Results that were diagnosed on the holdout

The rule above was broken once, in the open, and the numbers it touched are marked:
**every TypeSafe (Jev) result on this holdout from commit `7decebc` on is
diagnosed-on-holdout.** The fixes of that commit (pooled equivalent options, date
offsets and the "today" note, `keep_uncertain`, table-cell candidates, initials, the
identity span) were found by reading TypeSafe's live failures on holdout cases, and
the cases they fixed were then passing. A TypeSafe holdout number measured after that
commit is a dev number: do not quote it as blind, do not use it to choose a
combination, and do not compare it with a number measured before it. The offline
suite (scripted worlds) is unaffected by this, but is no evidence about a provider.

A fresh, blind holdout (`holdout2`) is written by someone who has not seen those
failures; use that one for any claim about TypeSafe, and apply the rule above to it
from its first run.

Where the library documents a limit, the gold follows the documented behaviour (and
the case's `note` says so): a relative edit ("two more seats") is refused with
`relative_edit_unsupported`, text values are verbatim spans, and an input that both
changes and removes a record is a `conflict_update_delete` (neither is applied).
