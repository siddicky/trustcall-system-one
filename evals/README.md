# Evals

Two suites share one case format (`evals/model.py`), one scorer and one runner.

| suite | file | cases | use |
| --- | --- | --- | --- |
| dev | `evals/cases.py` (`CASES`) | 29 | tuning: sweeps, threshold and prompt changes, choosing a provider combination |
| holdout | `evals/holdout_cases.py` (`HOLDOUT_CASES`) | 30 | measuring a combination that was chosen on the dev suite |

## The holdout suite

`HOLDOUT_CASES` is a second, independent dataset in the same format: new schemas,
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

Where the library documents a limit, the gold follows the documented behaviour (and
the case's `note` says so): a relative edit ("two more seats") is refused with
`relative_edit_unsupported`, text values are verbatim spans, and an input that both
changes and removes a record is a `conflict_update_delete` (neither is applied).
