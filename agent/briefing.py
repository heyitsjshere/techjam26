"""Domain briefing handed to the proposer every iteration.

SEEDING BOUNDARY -- read this before editing.

This file states FACTS ABOUT THE DATA AND THE TASK. It contains no measured
delta from Phase 1, no statement that any particular move works, and no final
configuration. Every fact here is one a competent practitioner would read
straight off the data or the organizers' own README in the first hour.

The structural mismatches are stated quantitatively and left unresolved on
purpose. Deriving candidate moves from them, and ranking those candidates, is
the agent's job and is what the Innovation axis scores.

If you are tempted to add "and chunking the groups fixes this" -- don't. That
is the answer, and handing over the answer makes the run log worthless on the
axis it is meant to demonstrate.
"""

TASK = """
You are optimising a recommender ranking on KuaiRand-Pure.

METRIC. Primary score = mean(GAUC, nDCG@5), computed by the organizers'
evaluate.py, which is authoritative and must never be reimplemented.
  * The task is WITHIN-USER ranking over each user's logged impressions. It is
    not full-catalogue retrieval. Only the relative order of a user's own rows
    affects the score.
  * nDCG@5 counts users with zero positives as 0.0 and keeps them in the mean.
  * GAUC counts only users with 0 < positives < impressions, weighted by
    positive count.
  * A consequence worth thinking through: any quantity that is CONSTANT across
    a given user's rows cannot change that user's ordering.

LABEL. long_view, binary. Positive rate is about 0.31 on both validation and
the training window.

BASELINE. The official baseline is a factorization machine trained with
POINTWISE logistic loss on five categorical fields (user_id, video_id,
author_id, tab, dur_bucket). Its validation primary is 0.6016. You must beat it.

CEILING. Perfect ranking scores 0.8484 on validation, not 1.0: 30.3% of
validation users have no positive label and 11.9% are all-positive, so only
57.8% are influenceable at all.
"""

DATA_SHAPE = """
SPLITS. train 20220408-20220421 (1,141,112 rows), valid 20220422-20220428
(124,909 rows). The test window is unreachable by construction.

GROUP SHAPE. This is a measured property of the data:
  * train: 26,210 users, 1,141,112 rows -> 43.5 rows per user (median 31, p90 97)
  * valid: 22,377 users,   124,909 rows ->  5.6 rows per user (median  4, p90 12)
  The lists you are scored on are roughly 6 items long. The lists present in the
  training window are roughly 43. Note also that nDCG@5 over a ~6-item list is
  close to "order the entire list correctly" rather than a top-of-list problem.

WHY THE SHAPE DIFFERS. From 22 April onward roughly 80% of impressions were
diverted into a randomised-exposure experiment logged separately, so the
standard log thins from ~81k rows/day to ~18k. This changes WHICH impressions
are logged, and how many each user has, in the evaluation window.

COVERAGE. 98.1% of valid users and 99.9% of valid videos also appear in train.
"""

ORGANIZER_FINDINGS = """
FROM THE ORGANIZERS' OWN README (their measurements, not ours):
  * Adding static feature fields is a measured dead end. They extended the five
    base fields to a 13-field set (music_id, video_type, upload_type, and six
    user-side demographic buckets) and the primary went from 0.5950 to 0.5940 --
    inside noise, slightly down.
  * Adding model capacity is a measured dead end. Embedding dimension 8 / 16 /
    32 gave 0.5895 / 0.5902 / 0.5887.
  * Their stated diagnosis: the user_id x video_id interaction already absorbs
    most of the learnable signal, and 1.14M rows will not support more capacity.
  * They also note that a purely user-side first-order term contributes exactly
    zero, because the ranking is done within a user.
  * The directions they list as UNEXPLORED, in their own order: (1) change the
    loss function, since the current one is pointwise but the metrics are
    ranking metrics; (2) user behaviour sequences, which no current feature
    uses; (3) multi-task use of the auxiliary feedback columns; (4) watch-time
    / duration-bias modelling; (5) different model families; (6) time features
    and distribution drift; (7) unbiased validation against the randomised log.

Treat all of the above as evidence to reason about, not as instructions.
"""

OBSERVATIONS = """
OBSERVATION FROM EARLIER MANUAL WORK, recorded without interpretation:

A collaborative-filtering feature (truncated SVD of the train-window user x item
long-view matrix, candidate scored by match to the user's profile) behaves
differently on the two evaluation surfaces. On the standard validation log it
scores worse than the same model without it. On the randomised-exposure
diagnostic it scores consistently better, across every encoding scheme tried.

That divergence is reported as an observation. What it implies about this
dataset, and whether it should change your candidate ordering, is yours to work
out.
"""

GUARDS = """
TWO HARD GUARDS. Both are enforced in code and will halt or reject; neither is
advisory.

1. OUT-OF-FOLD ENCODING ONLY. Any feature that aggregates the label or another
   outcome column over the training window must be out-of-fold encoded. Naive
   encoding lets a training row see its own outcome. Leave-one-out removes that
   but substitutes an offset that is a deterministic function of the row's own
   label, which a tree can learn to invert. Both are unreachable in agent mode.

2. MANDATORY DRIFT CHECK. Every feature block is checked for train-vs-valid
   distribution drift (standardised mean difference) BEFORE any metric computed
   on it is believed or recorded. A block that fails is rejected and the
   rejection is logged with its drift magnitude. Do not argue with the drift
   check by rerunning; diagnose the feature.

Also fixed and not negotiable: the randomised-exposure log is a read-only
diagnostic, never training data and never part of selection or convergence. The
test window is unreachable. is_rand is constant 0 and is excluded. The listed
same-row outcome columns are dropped at load time.
"""

METHOD = """
HOW THE HARNESS MEASURES YOU. These are properties of the measurement apparatus,
not advice about what will work.

A SPEC IS MEASURED AS A WHOLE, AND ATTRIBUTION DISCIPLINE APPLIES TO FEATURE
BLOCKS. The harness runs the configuration you submit and reports one score for
it. It does not decompose that score across the choices inside the spec. So if a
spec adds three feature blocks at once, a better or worse number tells you the
bundle moved the metric -- not which block did, nor whether one helped while
another hurt and the two partly cancelled.

That discipline matters most IN THE FEATURE SPACE, where individual effects are
small and mutually confounded and a bundled result is genuinely uninterpretable.

THE OBJECTIVE AND THE GROUPING ARE EXPLICITLY EXEMPT. Do not hold a structural
choice fixed in order to preserve feature-block attribution -- that is a
category error. Attribution discipline exists because small confounded effects
need isolating; a structural choice is neither small nor confounded with the
feature set, and freezing one to keep a clean feature comparison spends your
budget measuring the wrong axis. You may change the objective or the grouping at
any iteration, including while feature blocks also change, and you should not
treat an earlier spec as a reference configuration you are obliged to stay
near.

EVERY SPEC IS EVALUATED AT 3 SEEDS. You receive the mean and the standard
deviation. The baseline model's seed standard deviation is 0.0008, so a
difference between two configurations that is smaller than roughly that is not
distinguishable from seed noise, however suggestive it looks.
"""

CONVERGENCE = """
BUDGET AND STOPPING.
  * Hard cap 50 iterations. Wall-clock backstop 6 hours. A full train+evaluate
    cycle costs roughly 10-60 seconds, so compute is not your binding
    constraint; iterations and honest convergence are.
  * Convergence, exactly as this harness implements it. The organizers specify
    the constants (epsilon = 0.002, N = 3) but not the implementation, and their
    wording is ambiguous. We run the WINDOW reading:

        the run stops when   best_so_far(t) - best_so_far(t-3)  <=  0.002

    where t counts only iterations in which an experiment actually ran, and
    best_so_far is the best 3-seed mean validation primary achieved up to that
    point. Iteration 0 seeds best_so_far but is not itself a counting iteration.

    Two consequences follow directly from that formula, and you should reason
    about both. First, improvements ACCUMULATE inside the window: three
    successive iterations gaining 0.001 each sum to 0.003, which exceeds 0.002,
    and the run continues. (Those figures are arbitrary arithmetic to show the
    mechanism; they are not an estimate of what any move is worth.) Second, the window slides, so a gain only protects you
    for 3 counting iterations -- after that it leaves the window and stops
    counting toward the sum.

    The strict per-iteration reading (each single iteration must gain more than
    0.002 on its own) is computed in parallel and logged for disclosure, but it
    does not stop this run.
  * Because that rule can be satisfied by three unproductive iterations as
    easily as by genuine saturation, the ORDER you try things in determines
    whether you converge on a real ceiling or on your own choice of opening
    move. Rank your candidates by expected gain before you spend an iteration,
    and state the derivation.
  * Do not pad the run. Stopping honestly is scored; manufacturing marginal
    iterations is not.
"""


def full_briefing():
    return '\n'.join([TASK, DATA_SHAPE, ORGANIZER_FINDINGS, OBSERVATIONS,
                      GUARDS, METHOD, CONVERGENCE])
