# Phase 2 harness — design for review

Nothing here has been run as an agent dev run. Guards and plumbing are tested;
the loop itself is unexercised pending review of the seeding boundary and the
expected-gain ranking.

## Module map

| file | role |
|---|---|
| `src/firewall.py` | **unchanged from Phase 1.** Two locks; test window unreachable |
| `src/loader.py`, `src/metrics.py`, `src/diagnostics.py` | unchanged |
| `src/features.py` | feature blocks; encoder now calls Guard 1 on every encode |
| `agent/guards.py` | Guard 1 (out-of-fold only), Guard 2 (drift) |
| `agent/cache.py` | feature cache keyed by spec hash |
| `agent/runlog.py` | per-iteration structured log + run summary |
| `agent/actionspace.py` | Tier A schema and every available move. **No deltas** |
| `agent/briefing.py` | domain facts handed to the proposer. **No verdicts** |
| `agent/executor.py` | spec → metrics, with drift gating the metric |
| `agent/proposer.py` | LLM interface; candidate contract |
| `agent/controller.py` | gain-ranked scheduling, convergence, recovery |

## Guard 1 — out-of-fold only, unreachable not discouraged

`guards.check_encoding` is called from **inside** `_Encoder.encode`, so it fires
regardless of caller — including agent-written Tier B code that bypasses the
spec schema. In `agent_mode()` anything but `'oof'` raises `GuardViolation`,
which the controller never retries. The Tier A schema has no `encoding` field
and `validate()` rejects a spec that carries one.

Verified: `loo` and `naive` both raise in agent mode; `oof` works; schema
rejects an encoding field.

## Guard 2 — mandatory drift check, before the metric exists

Order in `Executor.run` is: build features → **drift check** → *only if passed*
train → score. A failing block returns `{ok: False, rejected_by: 'drift_check',
metrics: None}`. There is no code path that produces a metric for a
drift-failing block, so none can be recorded.

Statistic is standardised mean difference (scale-free, so one threshold covers
rates and log-durations). Thresholds were **calibrated empirically**, not
guessed:

| block | max SMD | |
|---|---|---|
| `duration` | 0.049 | legitimate |
| `item_agg` (out-of-fold) | 0.107 | legitimate |
| `cross_agg` (corrected) | 0.209 | legitimate |
| `dur_rank_in_list` | 0.268 | **covariate shift, not leakage** |
| `cross_agg` (naive) | **0.769** | known leak from Phase 1 |

Reject at **0.40** — 1.5× above the highest legitimate block, 1.9× below the
known leak. A **warn band at 0.25** exists because calibration surfaced a real
distinction: `dur_rank_in_list` is a duration percentile *within a user's list*,
and a percentile over 43.5 rows is not the same quantity as one over 5.6. That
is genuine covariate shift produced by the group-shape mismatch, not leakage.
Rejecting it would be a false positive, so it is flagged and passed. Every
feature's SMD is written to the log whether it passes or not.

## Cache

Key = sha256 over (block name, block implementation version, split fingerprint,
extra params), where the split fingerprint hashes users/dates/labels. A hit is
only possible when the produced frame would be identical. Version numbers let a
changed block invalidate its own stale frames. Hit/miss counts go into every
iteration record.

## Run log

One JSON object per iteration: `hypothesis`, `rationale`, `expected_gain`,
`expected_gain_derivation`, `spec`, `code_diff`, `drift_check` (full per-feature
report), `metrics` (GAUC / nDCG@5 / primary / per-seed / std), `diagnostic`
(`DIAG_*` unbiased metrics), `accepted`, `best_so_far`, `stall_count`, `error`,
`recovery`, `seconds`, `tokens_in`, `tokens_out`, `wall_clock_s`, `cache`.

Run summary: iterations used out of 50, **manual interventions**, total tokens,
agent wall-clock, GPU-hours (0), designated submission and reason, convergence
iteration and reason, git commit.

`record_intervention()` increments unconditionally. Instrumented from iteration
0 so the count is honest by construction rather than reconstructed.

## Seeding boundary

**Inherits — the full action space** (`actionspace.py`): both model families,
all three objectives, all six group-chunk values, all seven feature blocks
including `item_agg`, `user_agg`, `cross_agg` and `cf`, the full parameter grid,
train shaping, ensembling. Every move Phase 1 found dead is present and
selectable. Descriptions state what an action *does* and what it *costs*, never
what it is worth.

**Inherits — the domain briefing** (`briefing.py`): metric form and within-user
ranking; the consequence that a user-constant quantity cannot reorder a list;
~0.31 positive rate; baseline is pointwise FM at 0.6016; the 0.8484 ceiling;
train 43.5 rows/user vs valid 5.6, with medians and p90; the ~6-item list
consequence for nDCG@5; the 4/22 exposure regime change; coverage; the
organizers' README measurements including their two dead ends and their own
list of unexplored directions; and the CF-vs-diagnostic divergence **stated as
an observation with no interpretation attached**.

**Inherits — the two hard guards**, with the reasoning for each.

**Does NOT inherit:** any measured delta from Phase 1; the finding that group
chunking and `rank_xendcg` are the two moves that pay; the 2×2 interaction; the
final config; the drift magnitudes of specific blocks. The word "chunk" appears
in the action space only as a parameter with a neutral description.

The file headers state this boundary explicitly so it is not eroded by a later
edit.

## Expected-gain ranking — derived, not hardcoded

The controller has **no opinion** about which move is structural. It sorts
candidates purely by the proposer's own `expected_gain`. The mechanism that
makes that number mean something:

- Every candidate must supply `expected_gain_derivation` citing ≥1 fact key
  (e.g. `[GROUP_SHAPE_MISMATCH]`, `[POINTWISE_BASELINE]`). Fact keys are labels
  for briefing content, nothing more.
- A candidate citing no fact still executes if it ranks, but its gain is
  discounted by 0.5 and the discount is recorded. Rationale: an expected gain
  with no derivation is an assertion, not a prediction.
- The controller never ranks by *which* fact is cited. Whether the group-shape
  mismatch outranks feature work is the agent's call, made from the briefing,
  and visible in the log either way.

## Convergence hazard

The rule (eps 0.002, N 3) is implemented exactly as stated, measured against
**best-so-far** so a flat iteration after a real gain does not reset progress to
the previous iteration's value. Three mechanisms address the hazard without
hardcoding an ordering:

1. Expected-gain ordering, above — derived by the proposer, logged with its
   derivation.
2. **Forced high-variance move at `stall == 2`** — one iteration before the run
   would end, the proposer is told marginal variations are not acceptable and
   must propose the most different thing it can justify. This is the gain-aware
   scheduling in brief §6.5 and is the direct counter to converging on a poor
   opening move.
3. Best-so-far accounting, above.

The run is not padded and no minimum-iteration floor is imposed, because that
would be dodging the organizers' rule rather than scheduling around it.

## Iteration 0

The agent stands up the pipeline and reproduces the official FM baseline
itself, targeting validation primary 0.6016, logged in the same record
structure as every other iteration with `expected_gain: 0.0` and a rationale
stating it is a correctness precondition rather than an improvement attempt.
Costs ~19s. The FM is fitted against the **agent loader**, never the starter
kit's `data.load()`, which reads the test window.

## Recovery

- `GuardViolation` is never retried; the run halts. A hard guard firing means a
  correctness assumption is broken.
- Any other exception retries up to 2 times with the error text fed back to the
  proposer; after 2 failures the action key is marked dead and the controller
  routes to the next candidate.
- Proposal parse failures are logged as an iteration with `error` set and the
  loop continues.
- Best-so-far checkpoint is what gets designated, never the last iteration.

## Open item

The proposer backend is pluggable (`AnthropicBackend`, plus a `StubBackend` for
plumbing tests that the controller **refuses in a scored run**). Which backend
the scored run uses is unresolved — see the review note.
