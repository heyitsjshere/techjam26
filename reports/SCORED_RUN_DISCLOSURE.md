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

## Attempt 02 — see the results table
