# Scored runs — full disclosure

_Last updated 2026-08-30T16:41:42Z (UTC)._

Every scored run ever launched is listed here, including discarded ones, with
the reason. The pre-registered relaunch rule allows a relaunch on **technical
failure only**; a completed run with a disliked delta is explicitly not grounds.

## Attempt 01 — DISCARDED (technical failure)

- **Log:** `reports/runlog_scored_final_DISCARDED_01.jsonl` (retained, not deleted)
- **Launched at commit:** `e31177f`
- **Reached:** iteration 1 of 50. Iteration 0 reproduced the baseline at 0.60141
  (within the pre-registered ±0.0016 tolerance); iteration 1 ran and scored
  0.60050.
- **Outcome:** the process **crashed** during iteration 2's proposal.

### Cause

`client.messages.parse()` returns `parsed_output = None` when a response is
**incomplete** — typically `stop_reason='max_tokens'` — rather than raising. The
HTTP call succeeded, so this was not an API error and was not caught by any
error-specific handler. `slate.candidates` then raised `AttributeError`, which
propagated out of `propose()`, past the controller's `except ProposerError`, and
terminated the process.

This is a **harness bug**, not an agent result. The agent did not converge, did
not designate a submission, and produced no finishable run. It qualifies under
the technical-failure clause.

### Fix applied before relaunch

1. An incomplete response is now a **retryable** proposal failure with backoff,
   naming the `stop_reason`, rather than a `None` that flows onward.
2. `max_tokens` raised 16000 → 24000, since adaptive thinking plus the briefing
   can exhaust the smaller budget before the schema is emitted.
3. The controller now catches **any** proposer defect, not only `ProposerError`.
   The design rule was "an API failure must never kill the run"; the rule as
   implemented covered errors but not malformed successes.
4. `tests/test_liveness.py` gains two cases: a non-API proposer defect must
   degrade to the liveness abort rather than a traceback, and repeated
   incomplete parses must not end a run that later succeeds.

### Note on what this is evidence of

This is the **fourth** instance in this build of a guard that was correct in
isolation failing in composition (see F5, F8, F10, F11). The pattern is
consistent enough to be a finding in its own right rather than a run of bad
luck: each time, the mechanism did exactly what it was specified to do, and the
specification did not cover the case that actually occurred. Each was found only
by running the real system, never by tests written from the specification.

## Attempt 02 — DISCARDED (technical failure, introduced by the attempt-01 fix)

- **Log:** `reports/runlog_scored_final_DISCARDED_02.jsonl` (retained)
- **Reached:** iteration 0 reproduced the baseline at 0.60141. Iterations 1-3 all
  failed at the proposal step. The run then **stopped cleanly** via the liveness
  condition after 3 consecutive unproductive iterations.
- **Error:** `Streaming is required for operations that may take longer than 10
  minutes.`

### Cause

Fix (2) for attempt 01 raised `max_tokens` from 16000 to 24000. That crossed the
SDK's ceiling for **non-streaming** requests, so every proposal was rejected
before it was sent. The fix for one failure introduced another.

### What went right

The liveness condition added after the credit-exhaustion incident worked exactly
as designed: instead of spinning through all 50 iterations and reporting a
converged run, the run **aborted after 3 consecutive unproductive iterations and
named the underlying error**. Attempt 02 is the first failure in this build that
was caught by a guard written in response to an earlier failure, rather than by
a human reading the logs afterwards.

### Fix applied before relaunch

Switched from `client.messages.parse()` to `client.messages.stream()` with
`output_format`, which lifts the 10-minute ceiling while keeping the slate
schema-validated. Verified with a live call: 98s, 8,189 input / **7,871 output**
tokens — confirming that adaptive thinking plus the briefing genuinely needs
more than the original 16000 budget, which is what caused attempt 01's
incomplete response in the first place.

## Attempt 03 — DISCARDED (wrong model; configuration defect)

- **Killed deliberately** partway through, at iteration 0-1. No log retained
  because the run had produced no completed iteration beyond the baseline.
- **Cause:** `agent/run_agent.py` carried a hardcoded
  `--model default='claude-opus-4-6'`, written before the API reference was
  consulted, which **silently overrode** `proposer.DEFAULT_MODEL`
  (`claude-opus-5`). The CLI default won because it was passed explicitly to the
  backend constructor.

**Attempts 01 and 02 also ran on `claude-opus-4-6`**, confirmed from their
`proposer_model` fields. This is disclosed rather than quietly corrected: we had
documented and reported the proposer as `claude-opus-5` throughout, so the
harness was not doing what the documentation said, and the exact model string is
a required deliverable field.

The defect was invisible because the model string was faithfully logged —
`proposer_model: claude-opus-4-6` appears in every record of both discarded runs.
It was recorded correctly and simply never read back against the intended value.
A logged value nobody checks is not a control.

### Fix

`run_agent.py` now imports `DEFAULT_MODEL` from `proposer` rather than repeating
a literal, and `tests/test_designation.py` asserts the CLI hardcodes no model
literal, references `DEFAULT_MODEL`, and that `DEFAULT_MODEL == 'claude-opus-5'`.

## Attempt 04 — **COMPLETED. This is the scored run.**

- **Log:** `reports/runlog_scored_final.jsonl`
- **Commit:** `49f38e93caa9a54d51985eddaa79d39da6fbd8a5`
- **Proposer model:** `claude-opus-5` (verified in every record)

| | |
|---|---|
| Validation primary (3-seed mean) | **0.60209** |
| Delta vs official FM (0.6016) | **+0.00049** |
| Iterations used | 4 of 50 |
| Converged — window reading | iteration 3 |
| Converged — per-iteration reading | iteration 3 |
| Best-so-far curve | `[0.60141, 0.60208, 0.60209, 0.60209]` |
| Per-iteration seed std | 0.00004, 0.00057, 0.00022, 0.00045 |
| **Manual interventions** | **0** |
| Tokens (in / out / total) | 28,688 / 22,258 / 50,946 |
| Agent wall-clock | 444.0s (7.4 min) |
| GPU-hours | 0.0 |

**Designated submission:** `lightgbm · lambdarank · group_chunk=6 ·
[base5, dur_feats, item_agg]`, from iteration 2.

**Designation branch: `NO_DIVERGENCE`.** The best-valid config *is* the
structurally-justified one — both criteria selected iteration 2, 3-seed mean
0.60209, std 0.00022, against a band of 0.0008. `beats_baseline: true`.

This is the null result the rule was built to be able to produce. It was
executed, not asserted, and the record shows the two criteria agreeing rather
than us claiming they would have.

**Both convergence readings fired at iteration 3.** Consistent with every prior
run: the window reading has now extended zero runs out of ten. The agent
improved once (0.60141 → 0.60208), effectively plateaued (→ 0.60209), and
stopped. That null result is reported as evidence, not buried — a disclosed
choice that turned out not to matter is stronger than one that quietly helped.


---

## Test scoring — performed once, on 2026-08-31T01:22:28Z

Sequence enforced by the mechanism, not by intention: `--lock` ran first and
fingerprinted the submission **while the test score was still unknown**, then
`--score` verified the hash still matched before reading a single test label.

| | |
|---|---|
| Submission SHA-256 | `85c935323fdd4338e15149c2572b37e2bd4ab2e3a0428624b3c744c722898b62` |
| Locked at | 2026-08-31T01:22:17Z (commit `1599c7c6e5`) |
| Scored at | 2026-08-31T01:22:28Z |
| **Alignment mismatches** | **0** |
| Test GAUC | 0.661 |
| Test nDCG@5 | 0.52862 |
| **Test primary** | **0.59481** |
| Delta vs official FM test (0.5946) | **+0.0002** |

A second scoring attempt was made deliberately to confirm the one-shot marker
fires. It was refused.

The +0.0002 test delta is **smaller than the baseline's own 0.0008 seed std**. It
clears the gate; it is not distinguishable from a tie. Reported as such.
