# Evaluation policy, locked before Phase 1

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
bit-identical to the organizers' `data.load()`; the property `row_id`
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
on valid; the production recommender is worth a ~4× lift in long-view rate.
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
- **`is_rand`**, confirmed constant 0 across both standard logs in Phase 0.
 Carries no information. Dropped by decision.
- **`log_random` as training data**; the starter kit references it in zero
 lines of code; the README lists it only as an optional unbiased *validation*
 set. Not trained on.
- **No external training data. No pretrained weights.**

## 5. Manual-intervention accounting

Phase 1 is a human ceiling probe and is **not** a scored agent run; its
interventions are not counted. The intervention counter applies to the scored
run only, where the target is zero. Development runs and the scored run are
kept strictly separate, and that boundary is what makes the zero-intervention
claim true.

## 6. Submission designation rule, pre-registered

Written **before** the scored run, so it cannot be chosen after seeing which
config won.

**Rule.** If the best-valid config and the most-structurally-justified config
diverge, **the structurally-justified config is designated as final.**

A config is *structurally justified* when it was proposed to correct a stated,
measurable mismatch between the training setup and the evaluation setup, for
example a train/eval ranking-group-size mismatch, or an objective that does not
match the metric's form. A config is *merely best-valid* when its advantage
comes from the value of a hyperparameter, a seed, or a feature that was selected
because it scored well, with no mechanism named in advance.

**Rationale.** Phase 1 ran 79 experiments all selected on the same 124,909-row
validation split. Selection over that many candidates on one split buys
selection risk proportional to the number of candidates and to how flat the
landscape is, and this landscape is very flat: after the first two moves,
everything returned zero within a 0.0004 seed std. A structural fix does not
carry that risk in the same way; it is justified by a property of the data that
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

## 7. Iteration-0 baseline reproduction tolerance, pre-registered

Stated **before** the scored run so that "did the baseline reproduce" is a
numeric criterion, never a post-hoc judgement.

**Criterion.** Iteration 0 passes when the agent's validation primary is within
**2 seed standard deviations** of the official FM baseline:

```
|primary − 0.6016| ≤ 2 × 0.0008 = 0.0016      →  accept
```

0.0008 is the organizers' published 5-seed standard deviation
(`baseline_scores.json`). Two of them is a ~95% interval, so a correct pipeline
fails this at roughly a 1-in-20 rate on seed noise alone, tight enough to catch
a real pipeline defect, loose enough not to fail on chance.

**On failure the run halts.** It does not continue and it does not retry with a
different seed. A pipeline that cannot reproduce the baseline invalidates every
delta measured against it, so continuing would produce numbers that look fine
and mean nothing.

Reference points: the Phase 0 manual reproduction hit 0.6015 (deviation
−0.0001) and the harness smoke test hit 0.6014 (deviation −0.0002). Both pass
with large margin. Implemented as `BASELINE_TOLERANCE` in
`agent/controller.py`.

*Awaiting confirmation of the 2σ figure.*

## 8. Known limitation of the drift guard

Stated as a limitation rather than a feature, and repeated in
`reports/HARNESS_DESIGN.md`, because a guard described as complete is more
dangerous than no guard at all.

**The drift check is a backstop, not the primary defence.** The primary defence
against target leakage is out-of-fold encoding **enforced at the code level**
(`guards.check_encoding`, called from inside the encoder, so it also catches
agent-written Tier B code that bypasses the spec schema). That is the mechanism
that makes leakage structurally impossible for the aggregate features. The drift
check exists to catch what that enforcement does not cover; a *new* feature
family whose construction leaks by some route the encoder does not mediate.

**Its calibration band is narrow.** The observations span 0.049 to 0.268 for
legitimate blocks and 0.769 for the one known leak. Rejecting at 0.40 therefore:

- **catches gross leakage**, of the magnitude Phase 1 actually produced;
- **would miss a subtle leak** sitting anywhere around 0.30–0.40, and would very
 likely miss one below 0.27, which is inside the legitimate range;
- has **exactly one known-positive calibration point.** A threshold fitted to a
 single positive example is weakly determined, and we do not claim otherwise.

**It also cannot distinguish leakage from covariate shift.** Calibration
demonstrated this directly: `dur_rank_in_list` drifts at 0.268 for a legitimate
reason; it is a duration percentile computed *within* a user's list, and a
percentile over 43.5 rows is not the same quantity as one over 5.6. That is the
group-shape mismatch surfacing in the feature distribution, not a leak. The
0.25 warn band exists to make such cases visible rather than to adjudicate
them; a human reads them.

**What follows.** A drift *pass* is not evidence a feature is sound. It means no
gross leak was detected by a single-statistic test with one calibration point.
Any new feature family that aggregates outcomes still requires the mechanical
argument for why it cannot see its own row.

## 9. Convergence rule; an ambiguity, our reading, and the disclosure

Recorded **before** the scored run. We disclose this rather than quietly benefit
from it.

### The starter kit pins the constants, not the implementation

We checked. There is **no convergence code anywhere in the starter kit**:

- `baseline_scores.json` carries only `{"epsilon": 0.002, "N": 3}`; two numbers.
- `evaluate.py` is purely metrics; it has no notion of iterations at all.
- `baseline.py` has `patience=4` and an `eps`, but those are FM epoch
 early-stopping and Adam's numerical epsilon, different mechanisms entirely,
 not the agent-loop rule.
- The only prose is `README.md:72–73`.

### The prose is ambiguous, in Chinese as in English

> 连续 3 轮迭代 validation 主分提升不超过 0.002 即判定收敛。

Literally: "3 consecutive rounds of iteration [in which] validation primary
improvement does not exceed 0.002 → judged converged." 提升 ("improvement") is
unquantified as to whether it is *per round* or *cumulative across the three*.
The English in the task brief, "has not improved by more than eps over the last
3 consecutive iterations", carries the same two readings. This is not a
translation artifact; the ambiguity is in the original.

| reading | test | behaviour |
|---|---|---|
| **window** (ours) | `best(t) − best(t−3) ≤ 0.002` | three gains of 0.001 sum to 0.003 → continues |
| per-iteration | each of the last 3 gained ≤ 0.002 | three gains of 0.001 → converged |

### Our choice, and why

**We run the window reading.** Reasons, in order:

1. It is the more natural reading of "improved by more than eps **over the last
 N iterations**"; the phrase attaches the threshold to the span, not to each
 element of it. The per-iteration reading needs "in **each** of the last N".
2. The per-iteration reading makes N nearly inoperative. If every single
 iteration must clear eps on its own, the rule is "stop after 3 iterations
 that each individually failed to clear eps," and N is doing little work
 beyond a retry count.
3. It is measured against a 3-seed mean (§6), so seed noise cannot manufacture
 the accumulation.

**Disclosure.** The per-iteration reading is computed in parallel every run and
the iteration at which it *would* have fired is logged as
`converged_at_per_iteration` in every run summary and iteration record. The
results table reports the converged iteration under **both** readings. If the
organizers intended the strict reading, our run under it is reconstructable from
our own logs without rerunning anything.

**We note, without relying on it:** under the per-iteration reading eps = 0.002
is close to the entire attainable headroom we measured in Phase 1 (~0.0025), so
no sequence of genuine incremental gains can prevent convergence and only a
single large jump can. That is a property of the benchmark and is reported on
its own merits whichever reading is correct; it is not our argument for
choosing the window reading, which rests on the grammar and on point 2 above.

## 10. Iteration 0 does not start the stall counter

Iteration 0 reproduces the baseline. It **seeds** best-so-far and the
convergence window, but it is not an improvement attempt and does not count as
one. This is the same principle already applied to proposal failures and empty
slates: **only iterations in which an experiment actually ran are evidence about
saturation.** A correctness precondition is not evidence that the metric has
stopped moving.

Consequence: the first three *experiment* iterations form the first convergence
window, rather than the baseline consuming one of the three slots.

## 11. Clarification of §6, designation, disambiguated and made executable

**Timestamped 2026-08-30T15:03:29Z (UTC). Written before any scored run has been started;
none has.** This clarifies a clause that was internally ambiguous as written,
and corrects a policy/code mismatch. Both are disclosed rather than quietly
repaired.

### The ambiguity

§6 contains two clauses that can conflict:

- *"If the best-valid config and the most-structurally-justified config diverge,
 the structurally-justified config is designated"*, reads **unconditional**.
- *"Ties (within 1 seed std) resolve to the structurally-justified config"*,
 implies a **band**, and would be redundant if the first clause were truly
 unconditional.

Read together they do not determine what happens when the structural config is
worse by a wide margin. Taken literally, the first clause would designate a
structural config that lost by 0.05, which no one intended.

### The resolution; the conjunction

**Structural is designated when it diverges from best-valid AND is within one
seed standard deviation of it. Beyond that band, best-valid is designated and
the divergence is logged as a prominent record.**

    band = 0.0008                     # the organizers' published 5-seed std

    no structural candidate exists          -> record_intervention(), designate best-valid
    structural config IS best-valid         -> no divergence; designate it
    best_valid.primary - structural.primary <= band  -> designate STRUCTURAL
    otherwise                                        -> designate BEST-VALID, log divergence

The band is fixed at the **published** 0.0008 rather than at either config's own
measured std. A pre-registered constant cannot be tuned after seeing results; a
data-dependent band could be. Both configs' measured 3-seed stds are logged
alongside, so the alternative is reconstructable.

A config counts as **structurally justified** when the iteration that produced
it cited at least one of `GROUP_SHAPE_MISMATCH`, `POINTWISE_BASELINE`,
`USER_CONSTANT_NO_EFFECT`, `WITHIN_USER_RANKING`; the fact keys naming a
train/eval mismatch or a property of the metric's form, as opposed to a
parameter choice or a feature hunch.

### Why this needed fixing at all

`_designate()` returned best-valid unconditionally and stated the structural
rule inside its reason string as a promise that a human would apply it "at
designation review". That is a **policy/code mismatch**, and it makes the single
most consequential decision of the run an **unlogged manual intervention**. Our
zero-intervention claim was true of the loop and not true of the submission.

The rule is now executed in code. **There is no path where a human silently
decides**: if the rule cannot resolve a case, for example, no iteration cited a
structural fact, `record_intervention()` fires, the intervention counter
increments, and the reason says so. A run that needs a human to pick the
submission will report a non-zero intervention count, which is the honest
outcome.

### What a null result here means

If the designation record shows **no divergence**; the best-valid config is
also the structurally-justified one; that is not a wasted mechanism. It is the
useful output: the rule was executed and demonstrated rather than asserted, and
the record proves the two criteria agreed rather than leaving us to claim they
would have.

## 12. Policy audit, which clauses are executable, which are promises

Prompted by the §6 gap. Every clause in this document is classified below as
**CODE** (a mechanism enforces it, and a test proves the mechanism fires) or
**PROMISE** (a human keeps it; nothing in the harness can compel it). The
distinction matters because a promise written in the grammar of a mechanism is
worse than an honest promise; it invites everyone, us included, to believe
something is guaranteed when it is not.

| § | Clause | Status | Mechanism / test |
|---|---|---|---|
| 1 | Lock 1: test rows never enter the agent's process | **CODE** | `loader.load_agent` date-filters at parse; `test_firewall.py` |
| 1 | Lock 2: evaluate wrapper rejects test-window data | **CODE** | `firewall.assert_agent_safe`; breach test forges both routes |
| 1 | Selection and convergence use valid exclusively | **CODE** | no code path can construct a test split |
| 1 | Test scored **exactly once** | **CODE** *(was PROMISE; mechanized §14)* | one-shot marker `TEST_SCORED_ONCE.json`; second run refused |
| 1 | The test result cannot change the submission | **CODE** *(was PROMISE; mechanized §14)* | `--lock` fingerprints the submission before scoring; hash mismatch refused |
| 2 | Diagnostic clipped to the valid window | **CODE** | `diagnostics.load_unbiased_diag` asserts `max_date <= VALID_END` |
| 2 | Diagnostic never trained on | **CODE** | loader never includes it in train |
| 2 | Diagnostic never in selection or convergence | **CODE (by construction)** | no decision path reads `DIAG_*`; not separately asserted, see gap 4 |
| 3 | `evaluate.py` is the only scorer, never LightGBM ndcg | **CODE** | `metrics.score` wraps it; `metric='None'`; stopper calls `metrics.score` |
| 4 | Same-row outcome columns dropped at load | **CODE** | `firewall.assert_no_deny_columns`; tested |
| 4 | `is_rand` excluded | **CODE** | `DEAD_COLUMNS`, dropped in `_to_split` |
| 4 | `log_random` never trained on | **CODE** | never referenced by the train path |
| 4 | No external training data, no pretrained weights | **PROMISE** | nothing to enforce; none used |
| 5 | Dev and scored runs kept separate | **CODE** | `mode` flag, separate log files, `StubBackend` refused when scored |
| 5 | Every human touch increments the counter | **PROMISE** | `record_intervention()` must be called; a human who edits silently is not detected |
| 6 | Nothing single-seed is designated | **CODE** | `EVAL_SEEDS = [0,1,2]`, forced by the controller |
| 6 | Structural outranks best-valid, per the §11 conjunction | **CODE** *(was the gap; now fixed)* | `_designate()`; `test_designation.py`, 25 assertions, all five branches |
| 6 | Deltas from different origins are not summed | **PROMISE** | analysis discipline; Phase 1 addendum documents the one time we got it wrong |
| 7 | Iteration-0 tolerance, run halts on failure | **CODE** | `BASELINE_TOLERANCE`; `iteration_zero` returns False and `run()` finishes |
| 8 | Drift gate blocks the metric | **CODE** | `Executor.run` order: build → drift → *only if passed* → train |
| 8 | The guard's stated limitations | **DISCLOSURE** | not a mechanism; deliberately written as limits |
| 9 | Window reading runs; per-iteration logged in parallel | **CODE** | `_converged_window` / `per_iter_converged_at`; both in every summary |
| 10 | Iteration 0 does not start the stall counter | **CODE** | `ran_experiment=False`; window seeded, stall untouched |
| 11 | Designation conjunction and the intervention fallback | **CODE** | `_designate()` branches; `record_intervention()` on the unresolvable case |

### Was §6 the only gap? For that class of defect, yes.

§6 was the only place where **prose asserted a rule that the code actively
contradicted**. Every other CODE row was already enforced, and the audit found no
second instance of a mechanism claimed but absent.

But the audit did surface three weaker spots that were previously unstated. None
is a policy/code contradiction; all three are promises that read stronger than
they are, and they are recorded here rather than left implicit.

**Gap 2, "test scored exactly once" was unenforced. NOW MECHANIZED (§14).**

**Gap 3, "the test result cannot change the submission" was unenforceable as
written. NOW MECHANIZED IN THE FORM THAT MATTERS (§14):** the submission is
fingerprinted before the score is known, and a changed file is refused.

**Gap 4; the diagnostic's exclusion from selection is true by construction, not
by assertion.** No decision path reads `DIAG_*` keys, but nothing fails if a
future edit introduces one. *Mechanizable:* assert that decision inputs contain
no `DIAG_`-prefixed key. Low risk today, cheap insurance later.

**Gap 5; the intervention counter depends on being called.** `record_intervention()`
now fires on the one decision the harness makes that could otherwise be silent,
but a human who edits state mid-run is not detected. This is irreducible: the
counter measures honesty, it cannot manufacture it. What the harness can do; and
now does, is ensure the *agent loop itself* contains no unlogged human decision
point.

## 13. Iteration 0 is not a designation candidate

Added after a dev run exposed the omission in §11's implementation.

**Rule.** The designation candidate pool contains **experiment iterations only**.
Iteration 0 reproduces the official baseline as a correctness precondition; it is
not an attempt to improve on anything, and it is not eligible to be designated.
This is the same principle §10 already applies to the stall counter; only
iterations that ran an experiment are treated as evidence, extended to the
decision that selects the submission.

**What went wrong without it.** In a dev run where no experiment beat the
baseline, best-valid resolved to iteration 0 (the FM baseline itself, 0.60141).
The §6 within-band clause then compared it against a structural candidate at
0.60086, found the gap of 0.00055 inside the 0.0008 band, and designated the
**structural config, which scored below the baseline it exists to beat.** The
rule was executing exactly as written and produced a submission worse than doing
nothing.

**Additional requirement.** If the designated config does not beat iteration 0's
baseline, the designation record sets `beats_baseline: false` and the reason
carries an explicit warning. A run that produced no improvement reports that
plainly rather than presenting its best failure as a result.

**If no experiment iteration produced a metric at all**, there is no candidate,
`record_intervention()` fires, and nothing is designated.

This is a second instance of the F8 lesson: the rule was written down correctly
and implemented incompletely, and only executing it against real run data
surfaced the gap. Unit tests written from the policy text would not have caught
it, because the policy text did not mention iteration 0 either.


## 14. Test scoring: the last two promises, mechanized

Both clauses in §1 that the §12 audit found to be PROMISE are now CODE. These
were the ones a judge could reasonably doubt, because **the test labels are
sitting on our disk**; the discipline was self-imposed and, until now, only
self-attested.

**`--lock`** fingerprints the designated submission: SHA-256, byte size, row
count, timestamp, and the git commit, written to `reports/SUBMISSION_LOCK.json`.
It must be run **before** scoring. This is what makes "the test result cannot
change the submission" checkable rather than merely promised: the submission is
committed to while the test score is still unknown.

**`--score`** refuses unless a lock exists *and* the file still hashes to the
locked value. A submission edited after locking cannot be scored as though it
were the locked one. On success it writes `reports/TEST_SCORED_ONCE.json` and
**refuses every subsequent run**, naming the prior score and timestamp.

Neither mechanism can stop a determined human deleting a marker file. That is
not the claim. The claim is that the honest path is now the default path, that
any deviation requires a deliberate destructive act rather than an oversight,
and that both files are committed to git so a deviation is visible in history.

**Deliberately left as a PROMISE:** the diagnostic's exclusion from selection
(§2). It is true by construction; no decision path reads `DIAG_*`; and adding
an assertion would guard against a hypothetical future edit rather than any
present risk. It stays labelled honestly rather than dressed up.

## 15. Liveness condition

"An API failure must never kill the run" is a **safety** property. Composed with
"iterations that ran no experiment are not evidence about saturation", also
correct; it removed every condition under which a stalled run could stop for a
good reason. A run exhausted its API credits and passed all 50 iterations in 54
seconds, reporting itself converged.

The run now aborts after **3 consecutive iterations in which no experiment ran**,
naming the last error. This covers **every** unproductive path, not just API
failures: dead or invalid slates, and slates of only-repeats that survive the
reslate budget. `tests/test_liveness.py` exercises all three modes plus the
reset, confirming each aborts in under 12 iterations rather than at the cap, and
that an intervening productive iteration resets the counter.
