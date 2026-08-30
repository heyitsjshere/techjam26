# Evaluation policy — locked before Phase 1

Committed **before** the first Phase 1 experiment was run, and before the agent
existed. Timestamped by the git history of this file. Nothing below was decided
after seeing a result.

## 1. Test-set firewall

The hidden test window (20220429–20220508) is **present in the public
KuaiRand-Pure download**, inside `log_standard_4_22_to_5_08_pure.csv`. The
organizers' own `data.load()` slices it out with labels attached and
`baseline.py` prints test metrics. So test discipline here is **self-imposed,
not enforced by data availability**. We state that plainly rather than implying
the labels were withheld from us.

Two independent locks, implemented in `src/firewall.py`:

- **Lock 1 (structural).** `loader.load_agent()` discards every row dated past
  `VALID_END` during parsing. Test rows are never resident in the agent's
  process. There is no argument, flag, or config key that changes this.
- **Lock 2 (assertion).** Every `Split` carries its own date range. The agent's
  only scoring entry point, `metrics.score()`, asserts that range ends at or
  before `VALID_END` and that the split is not named `test`. A breach raises
  `FirewallBreach` and halts.

`tests/test_firewall.py` verifies both locks fire, verifies no `src/` module
imports the human-only path, and verifies our loader's row order is
bit-identical to the organizers' `data.load()` — the property `row_id`
alignment depends on.

**Policy.** Selection and convergence use `valid` exclusively. Test is scored
**exactly once**, by a human, via `src/human_only_test_scoring.py`, after the
agent has converged and locked its designated submission. Its sole purpose is
to detect `row_id` misalignment. **That result cannot change the submission,
the model, or any hyperparameter.** If it disagrees with valid, the
disagreement is reported as a finding, not acted upon.

## 2. Unbiased-exposure diagnostic

`log_random_4_22_to_5_08_pure.csv` is randomised-exposure traffic. Phase 0
established that ~80% of post-4/22 impressions were diverted into this
experiment, which is why the standard log thins from ~81k rows/day to ~18k.

It is wired as a **read-only per-iteration diagnostic**, clipped strictly to
20220422–20220428 to match the valid window; everything from 20220429 onward is
discarded at parse time because it overlaps test.

- Reported alongside valid every iteration, under a `DIAG_` prefix.
- **Not** in selection. **Not** in the convergence calculation. **Not** trained on.

Measured at wiring time: positive rate 0.0806 under random exposure vs 0.3133
on valid — the production recommender is worth a ~4× lift in long-view rate.
The diagnostic answers: does our ranking hold up on traffic the production
recommender did not choose?

## 3. Metric authority

Every experiment is scored by the organizers' `evaluate.py`, never by
LightGBM's internal `ndcg`. They disagree: LightGBM drops zero-positive groups,
`evaluate.py` scores them 0.0 and keeps them in the mean. 30.3% of valid users
are all-negative, so selecting on LightGBM's number would optimise a metric the
organizers do not compute.

## 4. Excluded from the action space

- **Same-row outcome columns** (`long_view`, `play_time_ms`, `is_click`,
  `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`,
  `profile_stay_time`, `comment_stay_time`, `is_profile_enter`) are dropped from
  every split's feature frame at load time. Historical aggregates over the
  **train window only**, via `loader.train_outcomes()`, are legal and used.
- **`is_rand`** — confirmed constant 0 across both standard logs in Phase 0.
  Carries no information. Dropped by decision.
- **`log_random` as training data** — the starter kit references it in zero
  lines of code; the README lists it only as an optional unbiased *validation*
  set. Not trained on.
- **No external training data. No pretrained weights.**

## 5. Manual-intervention accounting

Phase 1 is a human ceiling probe and is **not** a scored agent run; its
interventions are not counted. The intervention counter applies to the scored
run only, where the target is zero. Development runs and the scored run are
kept strictly separate, and that boundary is what makes the zero-intervention
claim true.

## 6. Submission designation rule — pre-registered

Written **before** the scored run, so it cannot be chosen after seeing which
config won.

**Rule.** If the best-valid config and the most-structurally-justified config
diverge, **the structurally-justified config is designated as final.**

A config is *structurally justified* when it was proposed to correct a stated,
measurable mismatch between the training setup and the evaluation setup — for
example a train/eval ranking-group-size mismatch, or an objective that does not
match the metric's form. A config is *merely best-valid* when its advantage
comes from the value of a hyperparameter, a seed, or a feature that was selected
because it scored well, with no mechanism named in advance.

**Rationale.** Phase 1 ran 79 experiments all selected on the same 124,909-row
validation split. Selection over that many candidates on one split buys
selection risk proportional to the number of candidates and to how flat the
landscape is, and this landscape is very flat: after the first two moves,
everything returned zero within a 0.0004 seed std. A structural fix does not
carry that risk in the same way — it is justified by a property of the data that
was measured *before* the experiment and holds independently of the split it was
scored on. Phase 1 also demonstrated the failure mode directly: the best single
seed (0.6046) beat the 5-seed mean (0.6041) by more than 2σ purely through
selection.

**Consequences, accepted in advance:**
- The designated submission may score lower on valid than some config we ran.
  That is the intended trade, not a mistake to be corrected later.
- Ties (within 1 seed std) resolve to the structurally-justified config.
- Any config accepted as an improvement must be confirmed at ≥3 seeds. Single-seed
  results are never designated.
- Marginal deltas measured from different origins are not additive and must not
  be summed to rank candidates. Interactions are measured, not inferred.
