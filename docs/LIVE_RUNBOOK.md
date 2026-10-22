# Live runbook

What to run against the real Clairic and Clef, in order, and which numbers to bring back.
Everything here is opt-in: `make test`, `make lint` and `uv run python -m evals run` never
touch the network. Credentials come only from environment variables; they are never written
to a file, a cassette or a report (the recording transport redacts them even if a body
echoes them).

## 0. Set-up

```bash
cd /home/user/trustcall-system-one
export CLOUDFLARE_ACCOUNT_ID=...      # from the environment only
export CLOUDFLARE_API_TOKEN=...       # (CLOUDFLARE_AUTH_TOKEN also works)
git rev-parse --short HEAD            # the commit under test: put it in the report
git status --short                    # must be clean; if not, say what is not
uv sync --all-extras --group dev
make lint && make test                # offline, must be green before anything live
uv run python -m evals run --quiet    # offline baseline: 24/24 on the scripted worlds
```

Use `UV_SYSTEM_CERTS` instead of the deprecated `UV_NATIVE_TLS` if uv warns. Do not print the
variables (`env | grep CLOUDFLARE` leaks the token); `test -n "$CLOUDFLARE_ACCOUNT_ID"` is enough.

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
grep -rlF -e "$CLOUDFLARE_ACCOUNT_ID" -e "$CLOUDFLARE_API_TOKEN" tests/recorded | wc -l   # must be 0
# the recordings replay offline through the same invariants:
uv run pytest --disable-socket tests/unit/test_recorded.py -v
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
| op precision, op recall | table | wrong operations, missed operations |
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
  `gates.add_threshold`, `gates.replace_threshold`, `gates.remove_threshold`,
  `gates.min_confidence`, `gates.min_choice_prob`, `limits.list_auto_below`, `match.accept`,
  `match.delete_threshold`, `coref.merge_threshold`.
- Change the requests (every point sends live): `limits.list_strategy`, `limits.sentence_follow`,
  `limits.max_sentences`, `gates.required_null_threshold` and `gates.optional_null_threshold`
  (sent to Clairic as `null_threshold`), `verify.instruction_template`, `verify.window_chars`,
  `verify.input_only_crosscheck`, `verify.attribution`, `passes.split`, `presence.*`,
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

## 5. What to hand back

1. The commit hash, the date, and whether the working tree was clean.
2. The smoke result (section 1) and the recorded file list (section 2).
3. The baseline table and the three strategy tables (section 3), with the failing cases.
4. The sweep tables (section 4) and the recommendations for the defaults above.
5. The paths of `evals/results/*` and `evals/cassettes/` (not committed) and of
   `tests/recorded/smoke_*.jsonl`.
6. Anything that surprised you: an HTTP error, a rate limit, a warning that appears in every run,
   a request count far from the offline one (offline: `uv run python -m evals run`, 190 requests for
   the 24 cases).
