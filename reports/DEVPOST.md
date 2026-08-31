# Devpost submission — An autonomous ML research agent for RecSys

**Repo:** https://github.com/heyitsjshere/techjam26

---

## Inspiration

We set out to build an agent that could run the whole MLE loop unattended on
KuaiRand-Pure. What we found is that the benchmark's own stopping rule makes
that nearly impossible to demonstrate, and the clearest evidence is a single run
of our own agent.

**In one run the agent improved its validation score on three consecutive
iterations — monotonically, with no wasted move — accumulating +0.00197 against
a convergence threshold of 0.002. It was stopped for *insufficient progress*,
0.00003 short. That is about one tenth of its own seed standard deviation.**

An agent that improved on every single iteration was terminated for not
improving. That is not a story about a weak agent. It is a property of a
stopping rule whose threshold is roughly 80% of the total headroom the benchmark
actually contains.

## What it does

An LLM-driven agent that reproduces the official baseline, forms its own
hypotheses about where signal might be, emits structured experiment specs, and
learns from the measured results — with **zero human interventions** during the
scored run.

| | GAUC | nDCG@5 | primary | delta |
|---|---|---|---|---|
| Official FM — validation | 0.6674 | 0.5357 | 0.6016 | — |
| **Agent — validation** | 0.6704 | 0.5378 | **0.60209** | **+0.00049** |
| Official FM — test | 0.6610 | 0.5282 | 0.5946 | — |
| **Agent — test** (scored once) | 0.6610 | 0.5286 | **0.5948** | **+0.0002** |

**4 iterations of 50 · 0 manual interventions · 50,946 tokens · 7.4 min · 0 GPU-hours**

We will be straight about the margin: **+0.0002 on test is smaller than the
baseline's own 0.0008 seed standard deviation.** It clears the baseline, but it
is not distinguishable from a tie, and we are not going to present it as a
decisive win. The findings are the substance here.

## How we addressed the problem statement

**A two-tier action space.** The LLM emits a structured spec; a deterministic
harness executes it. The agent never rewrites the pipeline, which is what keeps a
full autonomous run at ~51k tokens and 7.4 minutes.

**A seeding boundary we can prove.** Before the agent existed we ran a manual
probe of 79 controlled experiments. The agent inherits the *action space* —
every move, including all the ones we measured as dead ends — and a *domain
briefing* of structural facts. It inherits **no** measured delta, **no** verdict,
and **not** the final config. 72 automated assertions enforce that boundary, so
it cannot erode on a later edit. In every run the agent independently derived
train-group chunking from the stated 43.5-vs-5.6 group-shape mismatch, at
iteration 1 or 2. That derivation is in the run log, not in our briefing.

**Every policy enforced in code.** The test-set firewall is two independent
locks. Out-of-fold encoding is checked *inside* the encoder so agent-written code
cannot route around it. The drift gate sits between feature-building and
training, so no metric can exist for a leaking feature. The submission
designation rule executes rather than being described. **156 tests.**

**Honest by construction.** `reports/POLICY.md` classifies every clause as CODE
(a mechanism enforces it) or PROMISE (a human keeps it), because a promise
written in the grammar of a mechanism is worse than an honest promise. Three
scored attempts were discarded for technical failures and all three are
documented in full, including one where we ran the wrong model for three
attempts and disclosed it rather than quietly fixing it.

## How we built it

Python 3.13, LightGBM, numpy, pandas, scikit-learn. CPU only — no GPU, no torch.
The proposer is `claude-opus-5` via the Anthropic Messages API, with adaptive
thinking and structured outputs so a malformed proposal is impossible by
construction rather than caught by a parser.

## Challenges

**The benchmark is saturated, and we proved it three independent ways.** The
organizers' own ablations show more features and more capacity are both flat.
Swapping the model family entirely moves nothing — a pointwise GBDT scores 0.6017
against the FM's 0.6016. And a collaborative-filtering feature *hurts* validation
while consistently *helping* on randomised-exposure traffic, because validation
impressions were already preference-matched by a strong production recommender.
We are re-ranking a candidate set that a better-informed system already filtered.

**Four bugs were found only by running the real system**, never by tests written
from the specification. A duplicated `group_sizes` killed every chunked spec —
unit tests and a full stub run both passed, because the stub never ranked a
chunked candidate first. A designation rule, correctly specified and faithfully
implemented, designated a config *worse than the baseline*, because the policy
text never contemplated the baseline iteration as a candidate. A guard that
correctly refused to crash let a run spin through all 50 iterations proposing
nothing. And a model string was logged accurately in every record for three
attempts while nobody read it back.

## Accomplishments

A fully autonomous run that cleared the baseline with **zero interventions**, on
a benchmark already measured as saturated, under a stopping rule that halts an
agent improving on every iteration. Plus a policy document where every clause is
either mechanised or honestly labelled as a promise, and a disclosure record of
every discarded attempt.

## What we learned

**A pre-registered rule that is not executable is not pre-registration** — it is
an unlogged manual intervention in policy language. Ours returned the wrong
answer and deferred the real decision to "a human at designation review."

**Executable is necessary and not sufficient.** Once mechanised, the same rule
produced a submission worse than doing nothing, because the specification had a
hole and tests derived from that specification cannot find holes in it.

**A logged value nobody checks is not a control.** `proposer_model:
claude-opus-4-6` appeared in every record of three scored attempts, recorded
perfectly, read by no one.

**"Never fail" needs a liveness condition.** Composed with "iterations that ran
no experiment are not evidence about saturation," it removed every condition
under which a stalled run could stop for a good reason.

## What's next

Aggregates keyed on cold items, where the ID categoricals cannot already fit a
per-item effect. Real session boundaries instead of chunking by row count. And an
explicit argument to the organizers about the eps-to-headroom ratio: at 0.002
against ~0.0025 of attainable headroom, the rule selects for agents that guess a
whole configuration in one shot over agents that isolate variables — which is the
opposite of what the track is trying to reward.

---

## Built with

**Languages / libraries:** Python 3.13 · LightGBM 4.7 · NumPy 2.5 · pandas 3.0 ·
scikit-learn 1.9 · SciPy 1.18 · Pydantic 2.13

**APIs:** Anthropic Messages API — model `claude-opus-5`, adaptive thinking,
structured outputs, streaming

**Tools:** Claude Code (development) · git / GitHub · uv · Homebrew (`libomp`)

**Datasets:** KuaiRand-Pure, from https://kuairand.com via Zenodo record
10439422, under its own licence. Splits are the organizers' fixed date-based
ones. **No external training data and no pretrained weights were used.**

**Assets:** The organizers' KuaiRand-Pure Starter Kit, vendored unmodified in
`kuairand-starter-kit/`. `evaluate.py` is the sole scoring authority and was
never reimplemented.
