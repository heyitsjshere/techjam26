# Phase 1, manual ceiling probe: findings

79 logged experiments, ~26 min total train time, all scored by the organizers'
`evaluate.py` on **valid only**. Full record: `reports/phase1_log.jsonl`.

## Headline

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM baseline (valid) | 0.6674 | 0.5357 | 0.6016 |
| **Phase 1 best config, 5-seed mean** | **0.6704** | **0.5378** | **0.6041** |
| std over 5 seeds | 0.0004 | 0.0004 | 0.0004 |

**Isolated delta over FM: +0.0025 ± 0.0004.** Best single seed was 0.6046
(+0.0030) and is reported here only to be discarded, with a seed std of 0.0004
and 79 experiments selected on valid, quoting the best seed would be quoting
selection noise.

Locked config: `rank_xendcg`, train groups chunked to 6, features
`base5 + duration_ms + dur_feats`. That is 8 features and no aggregates.

## Isolated deltas; the seeded action space

Each row changes exactly one thing from the row it references.

| # | Move | primary | isolated Δ | verdict |
|---|---|---|---|---|
| 1 | LightGBM `lambdarank`, group=user (vs FM) | 0.5988 | −0.0028 | **worse than FM** |
| 1b | LightGBM `binary`, same features (control) | 0.6022 | +0.0006 | model class ≈ nil |
| | ⇒ objective delta, model held fixed | | **−0.0034** | pairwise ranking *hurts* |
| 2 | chunk train groups 43.5 → 20 / 10 / 7 / 6 / 4 rows | 0.6003–0.6021 | **+0.0033** | **real, largest single lever** |
| 3 | `lambdarank_truncation_level` 5…30 | 0.6017–0.6022 | ±0.0005 | no effect (lists ≤ 6) |
| 3 | `rank_xendcg` (listwise) vs `binary` | 0.6044 | **+0.0022** | **real** |
| 3b | grouping re-check under `rank_xendcg` | 0.6008–0.6044 | +0.0036 | chunk=6 confirmed |
| 4 | + item aggregates (video/author long-view & click rate) | 0.6039 | −0.0005 | **dead** |
| 5 | + user×item crosses (author, tag, duration pref) | 0.6042 | −0.0002 | **dead** |
| 6 | + duration features (log, within-list rank) | 0.6046 | +0.0002 | noise, kept |
| 7 | + user-level rates (long-view, click, like) | 0.6004 | −0.0040 | **delete** |
| 8 | target encoding: naive / **LOO** / OOF | 0.6041 / 0.5904 / 0.6039 | LOO **−0.0137** | see §Bugs |
| 8 | aggregates only, no ID categoricals | 0.5855 | −0.0189 | IDs subsume aggregates |
| 9a | drop `user_id` / `author_id` / both | 0.6036–0.6042 | −0.0004…−0.0010 | keep, but marginal |
| 9b | truncate train to last 10 / 7 / 4 days | 0.6017 / 0.5946 / 0.5831 | −0.0029…−0.0215 | **volume beats recency** |
| 9b | exponential recency weighting (0.5 / 1 / 2) | 0.6044–0.6047 | ±0.0002 | noise |
| 10–11 | + CF interest model (SVD, k=64), 3 encodings | 0.5634–0.5908 | −0.0138…−0.0412 | **dead on valid** (see §CF) |
| 12 | 7-axis hyperparameter sweep, 19 configs | 0.5958–0.6046 | best +0.0000 | **fully saturated** |
| 13 | rank-average ensembles, 5 variants | 0.6038–0.6045 | −0.0008…−0.0001 | **no gain** |

## Reasoning behind each outcome

**Why pairwise ranking hurts and listwise helps.** The organizers' top-ranked
untested idea was "switch to a ranking loss." Half right. `lambdarank` at
group=user is *worse* than pointwise, and only reaches parity once train groups
are cut to the size of evaluation lists. Its gradient is built from pairs
weighted by an NDCG position discount over the training list; at 43.5 rows per
group it spends that gradient on ranks 7–43, which no evaluation list ever has.
`rank_xendcg` optimises a listwise softmax with no position discount, so it is
far less sensitive to list length; and it is the only objective that beat
pointwise. Truncation level was irrelevant because chunking had already made
every list ≤ 6, which is the honest reading: **the truncation fix and the
chunking fix are the same fix**, and chunking is the one that generalises.

**Why every feature block is dead.** Not for the reason the organizers gave.
Their ablation added *static categorical fields* and concluded "features don't
help." The sharper statement, which step 8 measures in both directions, is that
**`video_id` as a 7.5k-level categorical strictly subsumes any train-window
aggregate keyed on video**. A tree fitting `video_id` directly learns a free
parameter per video from the same labels the aggregate averages; the aggregate
is a lossy compression of what the model already has. Adding it contributes
noise and a leakage channel. Dropping the IDs and keeping only aggregates costs
−0.0189, confirming the direction of subsumption. So aggregates are only worth
anything on keys the model *cannot* fit directly, which on this dataset is
nothing, because 99.8% of eval videos appear in train.

**Why user-level rates are worse than useless (step 7, decisive).** Ranking is
within-user, so a user-constant feature cannot reorder a list; it can only act
as a tree gate. Measured at −0.0040 with correct OOF encoding. It splits the
tree's capacity across user strata for no ordering benefit. **Deleted from the
action space. Do not re-add.**

**Why recency weighting does nothing but truncation hurts.** Valid sits
immediately after train and the exposure regime changes at 4/22, so recency
looked promising. It is flat, while hard truncation degrades monotonically with
how much data is removed (−0.0029 → −0.0215). Data volume dominates recency
here. The 4/22 regime change affects *which* impressions are logged, not the
user→item preference the model fits.

## Two methodology bugs found, and what they cost

Both were caught by the drift check (train-mean vs valid-mean per feature), not
by the metric. Both would have poisoned the agent's knowledge base with a false
feature verdict.

1. **Naive train-window aggregates leak.** An aggregate fitted on train and
 applied back to train rows contains that row's own outcome. `ua_affinity` had
 train mean 0.0149 vs valid 0.0003; a ~50× collapse; and the model fell to
 0.4866 primary, near random. Recorded initially as "cross features are
 catastrophic"; that verdict was **wrong** and was retracted.
2. **Leave-one-out is a worse fix than the bug.** Subtracting the row's own
 contribution removes self-inclusion but substitutes an offset of
 −1/(count+prior) that is a deterministic function of the row's own label. A
 deep enough tree learns "slightly lower rate ⇒ positive," which is inverted
 at evaluation time. Measured at **−0.0137** against naive. Out-of-fold
 encoding has neither defect and matches naive.

**Rule for the agent:** any target encoding must be out-of-fold. Leave-one-out
is banned. Every new feature is drift-checked (train mean vs valid mean) before
its metric is believed.

## The CF result, and why the diagnostic earned its keep

Collaborative filtering over the train-window long-view matrix; the organizers'
"completely blank" direction #2, the one thing not subsumed by `video_id`
because it is personalised *and* varies within a list, **hurts valid under all
three encodings** (−0.0138 to −0.0412), so it is rejected.

But it **raises the unbiased-exposure diagnostic every single time**, from 0.3664
to 0.3851–0.3942, a consistent +0.019 to +0.028.

That divergence is the most informative result in Phase 1. The mechanism:
impressions in valid were chosen by a strong production recommender that already
preference-matched the candidates, so within a valid list the CF signal is nearly
constant and contributes only variance. On randomly-exposed traffic the
candidates span the whole preference range and CF discriminates well.

**This explains why everything is flat.** We are re-ranking a candidate set that
has already been filtered by a better-informed system. The residual variance
inside that set is largely irreducible from the columns we are allowed to use.
It is also, independently, why the organizers' own feature and capacity
ablations came back flat.

Per policy the diagnostic cannot enter selection, and it did not: CF is rejected
on its valid number. It is recorded as a finding.

## Recalibrating the target

The brief's expected landing zone was +0.02 to +0.03 primary. **Phase 1 says
that is not reachable on this benchmark**, and the evidence is not one failed
attempt but a saturation pattern: objective, grouping, six feature families,
three encoding schemes, field ablations, distribution-shift handling, a 19-config
hyperparameter sweep across 7 axes, and 5 ensembles. Everything after the first
two moves returns zero within seed noise.

The realistic band is **+0.002 to +0.005**, and +0.0025 is already ~6σ against a
0.0004 seed std. Two implications for Phase 2, both in our favour:

- The **Feasibility gate** (hidden-test primary > baseline) is what matters, and
 +0.0025 on valid clears it with margin if it transfers. Transfer is the risk,
 not headroom.
- Since the model is saturated, **agent build quality is the whole remaining
 score**. Innovation (20%), Autonomy (20%) and robustness inside Technical
 Execution (35%) are where the points are. Grinding the model further is
 negative expected value.

## What the agent inherits

**Tier A action space, ordered by measured value:**
1. `objective` ∈ {`rank_xendcg` (best), `binary`, `lambdarank`}, measured spread 0.0034
2. `group_chunk` ∈ {4, 6, 7, 10, 20, none}, measured spread 0.0036
3. feature blocks ∈ {`base5`, `duration`, `dur_feats`}; the live set
4. hyperparameters, 7 axes, all measured flat; **low priority, not the first move**

**Dead moves, recorded with reasons so the agent does not re-derive them:**
`item_agg`, `cross_agg`, `user_agg`, `cf`, recency weighting, train truncation,
LOO encoding, ensembling, `is_rand`, and `lambdarank` at group=user.

**Guards Phase 1 proves are necessary:** a train-vs-valid drift check on every
new feature (it caught both bugs before the metric did), an out-of-fold-only
rule for target encoding, and multi-seed confirmation before any config is
accepted as an improvement.

---

# Addendum, closing the delta arithmetic (Item 1)

The Phase 1 marginals above (+0.0033 chunking, +0.0022 `rank_xendcg`) summed to
+0.0055 against a measured +0.0025. **Both marginals were single-seed**, against
a seed std of 0.0004, and worse, they were measured from **different origins**,
chunking from `lambdarank`/no-chunk (0.5988), `rank_xendcg` from `binary`/chunk=6
(0.6022). Neither origin was FM, so the two were never additive to begin with.

## The 2×2, 5 seeds per cell, common origin

Features held at `base5 + duration + dur_feats`.

| | chunk=None | chunk=6 | **effect of chunking** |
|---|---|---|---|
| `lambdarank` | 0.5994 ± 0.0002 | 0.6018 ± 0.0004 | **+0.0024** |
| `rank_xendcg` | 0.6009 ± 0.0003 | **0.6041 ± 0.0004** | **+0.0032** |
| **effect of xendcg** | **+0.0015** | **+0.0022** | |

Reference: `binary` 0.6017 ± 0.0004 (grouping is a no-op for a pointwise
objective). Official FM 0.6016.

## Resolution

- **Interaction = +0.0008.** The two moves are mildly *synergistic*, not
 independent: chunking is worth more under the listwise loss (+0.0032) than
 under the pairwise one (+0.0024). Recording two independent marginals would
 have mis-stated this in both directions.
- **Sum of marginals from the `lambdarank`/None corner = +0.0039; actual joint
 effect = +0.0046.** From a common origin the marginals *undershoot* the joint
 by the interaction term. The original +0.0055 was not this arithmetic at all.
- **The real source of the gap: the `lambdarank`/no-chunk corner sits 0.0022
 BELOW FM.** The joint move is worth +0.0046 measured from that corner, but
 0.0022 of it is spent climbing back to the baseline. Net versus FM:
 +0.0046 − 0.0022 = **+0.0024**, which is the measured +0.0025 within noise.

Single-seed inflation accounts for the rest: the chunking marginal drops from
+0.0033 to +0.0024 at 5 seeds.

## What the knowledge base records

Not "chunking is worth +0.0024 and listwise is worth +0.0022." Instead: **the
2×2 cell means with their interaction term, and the fact that the natural
corner to measure from (`lambdarank` at group=user) is itself below the
baseline.** A knowledge base holding two independent marginals would let the
agent rank a candidate by summing them, and it would be wrong by the interaction
term in one direction and by the sub-baseline origin in the other.

**General rule, now enforced in POLICY.md §6:** deltas measured from different
origins are not additive and must not be summed to rank candidates.
Interactions are measured, not inferred. Any accepted improvement is confirmed
at ≥3 seeds.

Corrected headline, unchanged: **+0.0025 ± 0.0004 over FM.** A pointwise GBDT
(`binary`, 0.6017) matches FM almost exactly, so the entire Phase 1 gain is
attributable to the listwise objective plus group-size matching, and to nothing
about the model family.
