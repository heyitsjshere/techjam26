# Results

## Headline

| | GAUC | nDCG@5 | primary | delta vs official FM |
|---|---|---|---|---|
| Official FM baseline — **validation** | 0.6674 | 0.5357 | 0.6016 | — |
| **Our agent — validation** (3-seed mean) | 0.6704 | 0.5378 | **0.60209** | **+0.00049** |
| Official FM baseline — **test** | 0.6610 | 0.5282 | 0.5946 | — |
| **Our agent — test** (scored once) | 0.6610 | 0.5286 | **0.5948** | **+0.0002** |

**Alignment mismatches on test: 0.** That check is the entire purpose of the
single permitted test scoring, and it passed.

### Read this honestly

The test delta of **+0.0002 is smaller than the baseline's own 5-seed standard
deviation of 0.0008.** It clears the baseline, so the Feasibility gate fires,
but it is not statistically distinguishable from a tie. We are not going to
present a within-noise margin as a decisive win.

The validation delta (+0.00049) transferred in sign and roughly in magnitude,
which is the more useful fact: nothing about the result collapsed between
validation and test, and the firewall held — the test set was read exactly once,
after the submission was hashed and locked.

## Designated submission

`lightgbm · lambdarank · group_chunk=6 · [base5, dur_feats, item_agg]`
3-seed mean, 140 boosting rounds. Selected at iteration 2 of the scored run.

**Designation branch: `NO_DIVERGENCE`** — the best-valid config and the
structurally-justified config were the same config. The pre-registered rule was
executed, not asserted, and produced its null result on the record.

## Resource report

| | |
|---|---|
| Iterations used | 4 of 50 |
| **Manual interventions** | **0** |
| Total tokens (in / out) | 28,688 / 22,258 = **50,946** |
| Agent wall-clock | 444s (7.4 min) |
| GPU-hours | **0** |
| Proposer model | `claude-opus-5` |
| Converged — window reading | iteration 3 |
| Converged — per-iteration reading | iteration 3 |

Three earlier scored attempts were discarded, all for technical failure, none
for a disliked delta. Fully documented in `SCORED_RUN_DISCLOSURE.md`.

## Context: what the headroom actually is

The organizers' own figures put perfect ranking at 0.8645 on test, because 27.1%
of test users have no positive label and 9.2% are all-positive — only 63.7% are
influenceable at all. Our own Phase 1 probe, 79 controlled experiments, measured
the *practically* attainable headroom above the FM baseline at approximately
**+0.0025**, and three independent lines of evidence agree the benchmark is
saturated (see `PHASE1_FINDINGS.md` F1).

Against that, +0.0002 on test and +0.00049 on validation are small numbers in a
small space, obtained under a stopping rule that permits roughly one experiment.
The findings, not the delta, are the substance of this submission.
