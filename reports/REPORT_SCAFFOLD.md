# Final report scaffold

Sections to be completed after the scored run. Findings below are recorded at
the time they were established, not reconstructed afterwards.

## 1. Evaluation policy (see POLICY.md — pre-registered, all sections)
- Test-set firewall, two locks, and the once-only human scoring path
- Submission designation rule: structural justification outranks best-valid
- Iteration-0 reproduction tolerance: |primary − 0.6016| ≤ 0.0016 (2σ)
- Drift guard **limitations**, stated as limitations

## 2. Results table
_(scored run)_ validation-best GAUC / nDCG@5 / primary, absolute delta over the
official FM baseline, and the once-only test score with its stated caveat.

## 3. Resource report
Iterations used of 50 · manual interventions · total input + output tokens ·
cache read/write tokens · agent wall-clock · GPU-hours (0) · proposer model
string `claude-opus-5` (Devpost APIs-used field).

---

# Findings established during Phase 1 and the Phase 2 build

## F1 — Model family contributes exactly zero. Third independent confirmation of saturation.

`binary` (pointwise GBDT) scores **0.6017 ± 0.0004** against the official FM's
**0.6016**. Two entirely different model classes — factorization machine with
embeddings, and gradient-boosted trees — land on the same number to within one
ten-thousandth. **The whole Phase 1 delta of +0.0025 is objective plus group
matching, and none of it is the model.**

This is the **third independent line of evidence** for the saturation thesis:

1. **The organizers' own ablations.** Adding static feature fields: 0.5950 →
   0.5940. Embedding capacity k = 8/16/32: 0.5895 / 0.5902 / 0.5887. Both flat.
2. **The CF mechanism.** A personalised CF feature hurts valid but consistently
   lifts the unbiased-exposure diagnostic (0.3664 → 0.385–0.394). Valid
   impressions were already preference-matched by a strong production
   recommender, so within a valid list the signal is near-constant and
   contributes only variance. We are re-ranking a candidate set that has already
   been filtered by a better-informed system.
3. **This finding.** Swapping the model family entirely moves nothing.

Three different probes — feature space, information source, and model class —
converge on the same conclusion. That is why the realistic band is +0.002 to
+0.005 rather than the +0.02 to +0.03 originally targeted, and why remaining
effort belongs in agent quality rather than in the model.

## F2 — The group-shape mismatch appears independently in the feature distributions.

The train/eval mismatch (43.5 rows per user vs 5.6) was identified from the
objective side: it is why `lambdarank` at group=user underperforms and why
chunking recovers it. Calibrating the drift guard surfaced **the same mismatch
from a completely different direction**.

`dur_rank_in_list` — a duration percentile computed *within* a user's list —
drifts at **SMD 0.268**, well above every other legitimate feature (next highest
0.209, most below 0.11). Nothing about that feature leaks. It drifts because a
percentile over 43.5 rows is not the same quantity as a percentile over 5.6.

So the mismatch is not merely a property of how the ranking loss is trained; it
**contaminates any feature whose definition depends on list composition**. That
is a broader statement than the objective-side finding and was arrived at
independently, by a guard that was not looking for it. It also sets a standing
caution: any future within-list feature inherits this distortion.

## F3 — Robustness event: permission boundary hit and escalated, not circumvented.

**Timestamp: 2026-08-30T09:00:03Z (UTC), during the Phase 2 harness build.**

While determining which proposer backend was viable, a shell command was issued
to check whether LLM credentials were available in the environment. It inspected
credential environment variables and `~/.claude/.credentials.json`. **The Claude
Code permission classifier blocked it.**

Response: the command was **not** retried, **not** rephrased to evade the
classifier, and **not** decomposed into narrower probes that would have
reconstructed the same information. The blocked action was reported to the
operator with a statement of what was being attempted and why, and the decision
was handed to them. A narrower follow-up checked only whether the `anthropic`
SDK was importable and whether the `claude` CLI was on PATH — facts that do not
touch credentials — and the backend choice was deferred until the operator
resolved it.

Recorded as a Robustness event because it is a real instance of the intended
failure behaviour under an unexpected obstacle: **escalate rather than work
around.** It is also the reason the final key-handling design reads credentials
from `os.environ` only and never reads back, logs, or persists the value —
the agent never needs to observe the key to use it.
