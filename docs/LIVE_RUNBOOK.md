# Live runbook

What to run against the real Clairic and Clef, in order, and which numbers to bring back.
Everything here is opt-in: `make test`, `make lint` and `uv run python -m evals run` never
touch the network. Credentials come only from environment variables; they are never written
to a file, a cassette or a report (the recording transport redacts them even if a body
echoes them).

## 0. Set-up

```bash
cd trustcall-system-one                # your checkout of the repository
export CLOUDFLARE_ACCOUNT_ID=...      # from the environment only
export CLOUDFLARE_API_TOKEN=...       # (CLOUDFLARE_AUTH_TOKEN also works)
# TypeSafe-only run instead: see section 4e (TYPESAFE_API_KEY and TRUSTCALL_S1_PROVIDER=typesafe)
git rev-parse --short HEAD            # the commit under test: put it in the report
git status --short                    # must be clean; if not, say what is not
uv sync --all-extras --group dev
make lint && make test                # offline, must be green before anything live
uv run python -m evals run --quiet    # offline baseline: 24/24 on the scripted worlds
```

Use `UV_SYSTEM_CERTS` instead of the deprecated `UV_NATIVE_TLS` if uv warns. Do not print the
variables (`env | grep CLOUDFLARE` leaks the token); `test -n "$CLOUDFLARE_ACCOUNT_ID"` is enough.
To run the live smoke test from GitHub Actions instead, see [CONTRIBUTING.md](https://github.com/siddicky/trustcall-system-one/blob/main/CONTRIBUTING.md#live-smoke-test-in-ci).

## 1. Smoke

```bash
make smoke 2>&1 | tee /tmp/smoke.txt
```

Runs four scenarios (`evals/smoke.py`): the README people conversation, a `One` insert, a
`One` update and a nested order update. Each asserts invariants only (no exception, every patch
applies, unmentioned fields untouched, nothing reaches another entity's record, spans quote the
input). It prints one line per scenario: requests, input tokens, records, warnings.

Report: pass or fail per scenario; for a failure the assertion message (it names the record
and the path); the per-scenario request and token counts; every warning printed.

## 2. Record

```bash
make record 2>&1 | tee /tmp/record.txt
ls -l tests/recorded/smoke_*.jsonl
# credentials must not be in the files (prints a count, not the value):
grep -rlF -e "$CLOUDFLARE_ACCOUNT_ID" -e "$CLOUDFLARE_API_TOKEN" tests/recorded evals/cassettes evals/results 2>/dev/null | wc -l   # must be 0
# the recordings replay offline through the same invariants:
uv run pytest --disable-socket --allow-unix-socket tests/unit/test_recorded.py -v
```

`make record` is `make smoke` plus `TRUSTCALL_S1_RECORD=tests/recorded`; it rewrites
`smoke_people.jsonl`, `smoke_invoice_insert.jsonl`, `smoke_contact_update.jsonl` and
`smoke_order_update.jsonl`. Do not commit them; list them and their sizes in the report so the
reviewer can. A replay that fails with `CassetteMissError` means the library's requests are not
deterministic between two runs of the same commit: report the kind and field it names, that is
a finding.

## 3. Evals baseline

The dataset is `evals/cases.py` (24 cases). Record into a cassette directory the first time, so
later runs and sweeps replay what has not changed:

```bash
mkdir -p evals/cassettes
uv run python -m evals run --live --config default \
    --record evals/cassettes --out evals/results/baseline-default
```

Then the three list strategies on the notes cases (and, to see the rest does not move, on all
cases). `--replay` + `--record` + `--live` together: recorded requests are replayed, the new ones
go to the API and are added to the cassettes.

```bash
for c in list sentences auto; do
  uv run python -m evals run --live --config $c --replay evals/cassettes --record evals/cassettes \
      --out evals/results/strategy-$c
done
for c in list sentences auto; do
  uv run python -m evals run --live --config $c --tag notes --replay evals/cassettes \
      --record evals/cassettes --out evals/results/notes-$c
done
```

`--config` names a preset (`default`, `list`, `sentences`, `auto`, `no_verify`, `strict`,
`lenient`; see `evals/configs.py`); `--set group.field=value` overrides one setting. `--concurrency N`
(default 4) is how many cases run at once; lower it if the API answers 429.

Each run writes `results.json` and `summary.md` into its `--out`. Report, per run, the first row
of `summary.md`:

| number | where | why it matters |
| --- | --- | --- |
| passed `x/24` | table | overall |
| op precision, op recall | table | wrong operations, missed operations (a legitimate extra note is an allowed extra: it is in neither) |
| notes recall | table | the expected notes that were produced, apart from the other operations |
| value accuracy | table | inserts: right values |
| bleed | table | a fact on the wrong record (`colleague_bleed`, `bleed_update`, `other_invoice_total`, `readme_people`) |
| wrong deletes | table | `delete_request`, `change_not_delete` |
| valid-record rate | table | records that validate |
| requests, input tokens, wall seconds | table | cost and latency per configuration |

and the failing cases with their `missing`, `extra` and `violations` text. Always include, from
`results.json`, the `probabilities` of `colleague_bleed` (Tom's Lisbon: live gave attribution
0.7736 against `attribution_keep` 0.75), `computed_total` (live 0.7695) and `other_invoice_total`:
the distance of those numbers from the thresholds is what the sweeps below move.

Cases that known-failed live at 05207aa, to confirm or clear: `readme_people` (Emma skipped as
`ambiguous_match`, no note added, Olivia invalid), `colleague_bleed` (Tom kept Lisbon),
`notes_update_contact` (no note appended).

## 4. Sweeps

`python -m evals sweep --param group.field=v1,v2,...` runs the grid (several `--param` make the
product; points that are invalid, such as `attribution_drop` above `attribution_keep`, are skipped
with a message) and writes `sweep.json` and `sweep.md`. A knob that only changes how already
received answers are judged replays from the cassettes: the table's last line says how many
requests each point sent live. A knob that changes the requests is announced on stderr and every
point needs live requests (which are recorded, so a second run is free).

Which knobs change requests (`evals/configs.py`, `REQUEST_LIGHT` / `REQUEST_CHANGING`):

- Replayable (the first-pass requests are the same; only the follow-up requests that depend on the
  decision, such as a re-ask in the middle attribution band, are new): `verify.attribution_keep`,
  `verify.attribution_drop`, `verify.low_confidence`, `verify.null_confirm_yes/no`,
  `verify.adjudicate_min`, `verify.clef_stated_min`, `verify.sentence_keep`,
  `verify.note_novelty` (only the novelty requests are new), `match.accept_compatible`,
  `discovery.record_presence` (only the record questions are new),
  `gates.add_threshold`, `gates.replace_threshold`, `gates.remove_threshold`,
  `gates.min_confidence`, `gates.min_choice_prob`, `limits.list_auto_below`, `match.accept`,
  `match.delete_threshold`, `coref.merge_threshold`.
- Change the requests (every point sends live): `limits.list_strategy`, `limits.sentence_follow`,
  `limits.max_sentences`, `gates.required_null_threshold` and `gates.optional_null_threshold`
  (sent to Clairic as `null_threshold`), `verify.instruction_template`, `verify.window_chars`,
  `verify.input_only_crosscheck`, `verify.attribution`, `verify.isolate_decisions` (one question
  per request or batched), `passes.split`, `presence.*`,
  `discovery.*`, `repair.*`.

The sweeps to run, all with `--live --replay evals/cassettes --record evals/cassettes`:

```bash
S="--live --replay evals/cassettes --record evals/cassettes"

# attribution: how much bleed is caught, how much correct data is lost
uv run python -m evals sweep $S --param verify.attribution_keep=0.65,0.7,0.75,0.8,0.85,0.9 \
    --out evals/results/sweep-attribution-keep
uv run python -m evals sweep $S --param verify.attribution_keep=0.7,0.75,0.8 \
    --param verify.attribution_drop=0.3,0.4,0.5 --out evals/results/sweep-attribution-grid

# the change gate, one threshold at a time
uv run python -m evals sweep $S --param gates.add_threshold=0.45,0.55,0.65,0.75 \
    --out evals/results/sweep-gate-add
uv run python -m evals sweep $S --param gates.replace_threshold=0.5,0.6,0.7,0.8 \
    --out evals/results/sweep-gate-replace
uv run python -m evals sweep $S --param gates.remove_threshold=0.75,0.85,0.95 \
    --out evals/results/sweep-gate-remove

# the open-ended list strategies
uv run python -m evals sweep $S --param limits.list_strategy=list,sentences,auto \
    --out evals/results/sweep-list-strategy
uv run python -m evals sweep $S --param limits.list_strategy=sentences,auto \
    --param verify.sentence_keep=0.5,0.6,0.7,0.8 --tag notes \
    --out evals/results/sweep-sentence-keep
uv run python -m evals sweep $S --param limits.list_strategy=sentences \
    --param limits.sentence_follow=0,2,4,6 --tag notes \
    --out evals/results/sweep-sentence-follow
uv run python -m evals sweep $S --param limits.list_strategy=auto \
    --param limits.list_auto_below=0.1,0.5,0.9 --tag notes \
    --out evals/results/sweep-auto-below
```

Add `--tag bleed` to a sweep to see only the cases that test a guard (`colleague_bleed`,
`bleed_update`, `other_invoice_total`), `--tag update` for the update path only.

What to report from each sweep, from its `sweep.md` table (one row per point):

- passed `x/24`, op precision, op recall, value accuracy, bleed, wrong deletes, valid rate,
  requests, input tokens, wall seconds, and the live requests per point.
- The best point and why: the setting with the highest recall at zero bleed and zero wrong
  deletes; where recall starts to fall (a threshold that is too strict) and where bleed starts to
  appear (too lenient). The current defaults are `attribution_keep` 0.75, `attribution_drop` 0.4,
  `add_threshold` 0.55, `replace_threshold` 0.6, `remove_threshold` 0.85, `sentence_keep` 0.6,
  `sentence_follow` 4, `list_auto_below` 0.5; say whether each should change.
- For the strategy sweeps: per notes case, which strategies appended the notes (`notes_update_contact`,
  `notes_insert_contact`, `notes_people_insert`, `readme_people`), how many extra requests
  `sentences` and `auto` cost over `list` (requests and input tokens), and whether any note is
  attributed to the wrong person (`notes_people_insert`).
- Any case with `error` set (a transport error, a `CassetteMissError`): the message.

## 4b. Clef on Experiential Labs

Clairic stays on Cloudflare; only the Clef requests go to Experiential Labs
(`POST https://api.experientiallabs.ai/v1/systemone`). Everything above works unchanged with
the provider switched.

```bash
# in addition to the Cloudflare variables of section 0 (from the environment only):
export EXPERIENTIAL_API_KEY=...                  # never write it to a file; do not echo it
test -n "$EXPERIENTIAL_API_KEY" && echo set      # enough to check it
export TRUSTCALL_S1_CLEF_PROVIDER=experiential   # required: the key alone never moves Clef

make smoke 2>&1 | tee /tmp/smoke-experiential.txt
uv run python -m evals run --live --clef-provider experiential --config default \
    --replay evals/cassettes --record evals/cassettes --out evals/results/experiential-default
```

Which provider serves Clef: `--clef-provider` (evals only), else `TRUSTCALL_S1_CLEF_PROVIDER`
(`cloudflare` or `experiential`), else Cloudflare: an `EXPERIENTIAL_API_KEY` alone never moves
Clef, because that would send the text to another company. `--clef-provider` without `--live`
only prints a note. `results.json` records the provider as `clef_provider` and `summary.md`
prints it, for a live run and for a replay (which sends only what the cassettes lack). Cassettes
are keyed by the model id and the request body, so a cassette recorded on Cloudflare replays
under Experiential and the other way round; to compare the providers fairly, record into
separate directories (`--record evals/cassettes-experiential`), or leave `--replay` out.

No request to Experiential is ever retried once it may have reached the server. A timeout, a
dropped connection or a 5xx fails the case with `UncertainOutcomeError` after exactly one
request, and that call may have been charged. In the report:

- count the calls that may have been charged without an answer: `unknown_outcomes` in
  `results.json` (per case and in `aggregate`) and the "Calls with an unknown outcome" line of
  `summary.md`. It counts every call that ended in `UncertainOutcomeError` and every call that was
  *cancelled after it may have been sent* (when one request of a case fails, its siblings that
  are on the wire are cancelled; they are logged with `error` "cancelled after send; outcome
  unknown" and have no response). The `requests` of a failed case now counts those calls too.
  For the uncertain ones give each message, with the `correlation id` it carries when the server
  sent one: that is what to quote when asking Experiential Labs whether a call was charged;
- the cost is the `reported_cost_usd` of `aggregate` (what Experiential reports per call, only for
  the calls that answered); calls without an answer are not in it, which is one more reason to
  report the unknown outcomes;
- a 429 storm can use up the 300 s budget: a re-send is not made when less time is left than an
  attempt needs (`min_attempt_budget`, default the 60 s read timeout), and the call fails with a
  deadline error saying the request was not sent, instead of sending a request that is likely to
  be cut off and charged;
- do not re-run just those cases to "fix" the table without saying so: each re-run is a new
  charge. A 429 with `Retry-After` and a failed connection are retried inside the transport and
  appear as `retries` in the call log only;
- start with `--concurrency 2` and raise it only while there is no 429.

The response shape was confirmed by a live probe and is saved in
`docs/schemas/experiential-clef-response.example.json` (see `docs/DESIGN.md`, Transport): a bare
Clef response plus `id`, `provider` and `usage.cost`. `usage.cost` is the price of the call as
Experiential Labs reports it, and the library sums it as `usage.reported_cost_usd`
(`result.cost.reported_usd`), apart from the token-price estimate. Before committing anything,
run `grep -rnE "xpl_[A-Za-z0-9_-]{16,}" . --exclude-dir=.git`: it must find nothing.

## 4c. Decisions API as the Clef fallback

`DecisionsTransport` answers Clef requests with the Decisions API
(`POST https://api.experientiallabs.ai/v1/decisions`, model `gpt-6-luna-decisions`, the same
`EXPERIENTIAL_API_KEY`), and `FallbackTransport` uses it when the Clef provider (Experiential or
Cloudflare) could not answer. The Decisions API is served by OpenAI through Experiential
Labs, so the fallback sends text to both: it is opt-in:

```bash
export EXPERIENTIAL_API_KEY=...                  # from the environment only
export TRUSTCALL_S1_CLEF_FALLBACK=decisions      # required: a key alone enables nothing
# default: fall back only when the primary certainly ran nothing (a 429, a failed
# connection). A timeout or 5xx may have been charged, and the fallback asks the same
# question again, so it is a separate opt-in (the question can be paid for twice):
# export TRUSTCALL_S1_CLEF_FALLBACK_ON=unavailable,uncertain,bad_response
make smoke 2>&1 | tee /tmp/smoke-fallback.txt
uv run python -m evals run --live --clef-provider experiential --clef-fallback decisions \
    --config default --out evals/results/fallback-default
# with the uncertain outcomes too (read the second bullet below first):
#   ... --clef-fallback decisions --clef-fallback-on unavailable,uncertain
```

How to read it:

- A request the fallback answered has `provider` `decisions` and `fallback_from` in its call
  record, the result carries the warning `clef_fallback_used: N Clef request(s) ...`, and
  `summary.md` prints "Clef provider: experiential + fallback decisions". Report N per case. A
  run with a clean primary should have none; every one is a primary failure to explain.
- The primary's failed attempt is a record of its own, with its provider, and so is the
  fallback's when it fails too (error, cost, cancellation): `unknown_outcomes` counts a primary
  timeout plus a fallback timeout as 2, a primary timeout plus a bad charged response as 1, and a
  primary timeout plus a cancelled fallback as 2. By default the fallback is used only when the
  primary certainly ran nothing, so a clean default run has no uncertain primary records; if
  you opted in to `uncertain` each one of them may be a second charge for the same question.
  Fallback calls are filed under `gpt-6-luna-decisions` in `usage.by_model` (its reported cost is
  in `reported_cost_usd`; Clef's pricing is not applied to them).
- A field the fallback answered has `provider` `decisions` and the note `answered_by:decisions`
  (`hard_answer` for a bare pick), and a Clairic field it checked has `attribution_by:decisions`
  in `warnings`; the `clef_fallback_used` warning lists up to 8 uses.
- The fallback is a different model: its answers are not Clef's. Observed probabilities were
  saturated (0.0 and 1.0, confidence 1.0), which is a bare pick: it is tagged `hard` and capped at
  0.8 (`decisions_hard_cap`), below the 0.85 remove / set-null gate, so a fallback answer cannot
  authorise a removal by itself (an add or a replace still passes) and the re-ask bands do not
  fire on it.
  Compare the quality of a fallback run against a normal one before relying on it; a case that
  only passes because the fallback answered is a finding, not a pass.
- `images` cannot be sent to the Decisions API: such a request fails with the primary's error.
- Never re-run only the failed cases quietly: each request that reached either provider may have
  been charged.

## 4d. The Decisions API as the primary for Clef

```bash
export EXPERIENTIAL_API_KEY=...                  # from the environment only
export TRUSTCALL_S1_CLEF_PROVIDER=decisions      # required: a key alone enables nothing
uv run python -m evals run --live --clef-provider decisions --config default \
    --out evals/results/decisions-primary
```

The Decisions API (`gpt-6-luna-decisions`, served by OpenAI through Experiential Labs) answers
every Clef request; Clairic stays on Cloudflare. It takes 32 questions per request and its bare
picks are capped at 0.8 (`decisions_hard_cap`), so compare it with the default run before
trusting it. It is a different model from Clef: a case that only passes with it is a finding.
`summary.md` prints "Extraction: cloudflare clairic; decisions: experiential decisions
(gpt-6-luna-decisions, hard cap 0.8)", and `results.json` has the same as `extraction` and
`decisions`.

Reading a Decisions-primary run:

- The warning is `clef_hard_picks: N Clef answer(s) from decisions were hard picks ...; by call
  kind: ...` (saturated or missing probabilities, capped at 0.8). It is not `clef_fallback_used`,
  which is only for a request a fallback answered and is absent here.
- A `decisions_refusal` warning names the questions the model refused (`{"type": "refusal"}`):
  they are unanswered, never a pass, and the case goes on.
- A delete needs a graded 0.85 or more (a saturated yes is 0.8 by design), so expect fewer
  deletes than with Clef: `visit_cancel_one`, `listing_no_longer_pets` and
  `order_remove_item` were such cases. An input that removes AND changes a record may be read as
  a plain change (no `conflict_update_delete`): that expectation is provider-specific.
- Notes: a notes field is classified (call kind `list_kind`); a hedged `items` below 0.75 no
  longer keeps notes out, so a Decisions run has `sentences` and `novelty` calls where the first
  live run had none.

## 4e. TypeSafe-only: Jev for Clairic and Clef

```bash
export TYPESAFE_API_KEY=...                      # from the environment only; do not echo it
test -n "$TYPESAFE_API_KEY" && echo set          # enough to check it
export TRUSTCALL_S1_PROVIDER=typesafe            # required: the key alone never switches
# no Cloudflare variable is needed

# live smoke: the four scenarios of section 1 on Jev
TRUSTCALL_S1_LIVE=1 TRUSTCALL_S1_PROVIDER=typesafe uv run pytest -m live -s tests/live
# the evals on Jev (start with --concurrency 2; raise it while there is no 429)
uv run python -m evals run --live --provider typesafe --config default --concurrency 2 \
    --out evals/results/typesafe-default
# the examples
uv run python examples/01_invoice.py
```

In the report, for each case or for the totals: the requests (a Clairic call is a stitch request, a
selection request and a verification request, so expect about twice the requests of the Cloudflare
run), the input tokens (the state is repeated in each; Jev's input is $0.042 per Mtok and its output is
free), the wall seconds, and which warnings fired. Per call the log has `provider` `typesafe` and
`model` `jev-<version>` (the model that answered; `jev-latest` moves). Read these first:

- `escalated` in a field's `raw` answers (the verification doubted a selection) and `needs_review` on a
  date. Report how many selections were doubted and whether the ladder rejected them rightly.
- A case whose expected value is not among the candidates can never pass (Jev only selects): report the
  value and the sentence, that is a finding for the candidate generators, not for Jev.
- Calibration: the library's thresholds were tuned on Cloudflare. Report the attribution
  probabilities (`probabilities` in `results.json`) next to the Cloudflare run before changing any.
- An `UncertainOutcomeError` (a timeout, a 408 or any 5xx including 529 Overloaded, never resent: the
  call may have been charged; a Clairic call that fails after an earlier request answered keeps those
  tokens in the call log and counts as uncertain), a `RateLimitError` after the retries (only 429 is
  retried, with backoff), and a 422 (the message names the field; the body and the user's text are not
  echoed).
- `needs_review` on a date whose year or week was inferred (confidence capped at 0.8: those are re-asked),
  and `escalated: hook_error` / `hook_invalid` if you passed an escalation hook.
- Before committing anything run `grep -rnE "apikey_[0-9a-f]{20,}" . --exclude-dir=.git
  --exclude-dir=.venv`: it must find nothing.

To tabulate the three combinations (Cloudflare, Cloudflare Clairic + Decisions Clef, Jev for both) run
sections 3, 4d and 4e into separate `--out` directories and compare `aggregate` (`passed`, `requests`,
`input_tokens`, `wall_s`) and the `extraction` / `decisions` labels of each `results.json`.

## 4f. Mixing providers per call kind and field type

Everything in this section is opt-in and names its providers: a spec file says which provider
extracts which field type and answers which call kind (`docs/DESIGN.md`, "Mixing providers per call kind
and field type"). Offline first (scripted worlds, nothing is sent), then live:

```bash
uv run python -m evals combo --spec spec.json --quiet                    # offline
uv run python -m evals combo --spec spec.json --live --record evals/cassettes --tag NAME \
    --out evals/results/combo-NAME            # cassettes go to evals/cassettes/NAME
uv run python -m evals search --base spec.json --live --providers cloudflare,typesafe,decisions \
    --price cloudflare=P:P --price typesafe=P:P --price decisions=P:P --max-runs 40 --budget-usd 10 \
    --record evals/cassettes --out evals/results/search     # dev suite only
# calibration of the winner: record it once, then search the shifts over the raw recording
uv run python -m evals combo --spec evals/results/search/best.json --live --record evals/cassettes --tag BEST
uv run python -m evals search --base evals/results/search/best.json --calibrate-only \
    --replay evals/cassettes/BEST --out evals/results/search-cal         # --live: misses go to the API
uv run python -m evals combo --spec evals/results/search-cal/best.json --suite holdout --live
```

A failed case is reported, never retried; a trial that raises is a row in the leaderboard and the search
goes on; `--resume DIR` reuses the combinations already evaluated there IN THE SAME MODE (an offline
search resumed with `--live` reuses nothing). `--replay` is for the calibration search alone, and a
recording only answers the provider that made it: a spec that routes a call elsewhere gets a miss, never
another provider's answer. A positive shift of `gate`, `null_check` or `delete` raises the probability that
decides a removal: the search leaves those kinds out unless they are named in `--calibrate-kinds`, tries
only shifts of 0 or less for them unless `--allow-positive-removal-shift`, and flags them in the
leaderboard. An improvement has to beat the incumbent by `--min-margin` (default 1 case or 0.02 F1).
Hand back `leaderboard.md`, `best.json` and the holdout result of the winner (its directory is named
`...-holdout-...`).

## 5. What to hand back

1. The commit hash, the date, and whether the working tree was clean.
2. The smoke result (section 1) and the recorded file list (section 2).
3. The baseline table and the three strategy tables (section 3), with the failing cases.
4. The sweep tables (section 4) and the recommendations for the defaults above.
5. If the Decisions fallback was on (section 4c): the number of fallback answers per case, the
   warnings, and which primary errors caused them.
6. If Clef ran on Experiential Labs (section 4b): the provider, the count of uncertain
   outcomes with their messages and correlation ids, and any response that differs from the
   saved example.
7. The paths of `evals/results/*` and `evals/cassettes/` (not committed) and of
   `tests/recorded/smoke_*.jsonl`.
8. Anything that surprised you: an HTTP error, a rate limit, a warning that appears in every run,
   a request count far from the offline one (offline: `uv run python -m evals run`, 190 requests for
   the 24 cases).
