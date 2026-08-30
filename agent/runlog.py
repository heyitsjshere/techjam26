"""Per-iteration run log. A graded deliverable, so it is first-class.

One JSON object per iteration carrying everything section 8 of the brief asks
for, plus the two things Phase 1 proved you need: the unbiased-exposure
diagnostic, and the drift-check outcome recorded BEFORE the metric is believed.

The manual-intervention counter is instrumented from the first iteration.
Every human touch increments it, with no exception, because reconstructing an
honest count afterwards is not possible.
"""
import json
import os
import subprocess
import time


class RunLog:
    def __init__(self, path, run_id, mode='dev', model=None):
        self.path = path
        self.run_id = run_id
        self.mode = mode                 # 'dev' or 'scored'
        self.model = model               # exact model string, a Devpost field
        self.cache_read = self.cache_write = 0
        self.t0 = time.time()
        self.interventions = 0
        self.tokens_in = self.tokens_out = 0
        self.iterations = 0
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # -------- intervention accounting --------
    def record_intervention(self, what, who='human'):
        """Any human touch during a run. Increments unconditionally."""
        self.interventions += 1
        self._write({'kind': 'INTERVENTION', 'n': self.interventions,
                     'what': what, 'who': who, 'wall_clock_s': self._elapsed()})

    # -------- per-iteration --------
    def iteration(self, *, i, tier, hypothesis, rationale, expected_gain,
                  expected_gain_derivation, spec, code_diff=None,
                  drift=None, metrics=None, diagnostic=None, accepted=None,
                  best_so_far=None, stall_count=None, error=None, recovery=None,
                  seconds=None, tokens_in=0, tokens_out=0, cache=None,
                  usage=None, api_recovery=None, extra=None):
        self.iterations = max(self.iterations, i + 1)
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if usage:
            self.cache_read += usage.get('cache_read_input_tokens', 0) or 0
            self.cache_write += usage.get('cache_creation_input_tokens', 0) or 0
        rec = {
            'kind': 'ITERATION', 'run_id': self.run_id, 'mode': self.mode,
            'iteration': i, 'tier': tier,
            # --- Innovation axis: what it chose to try, and why ---
            'hypothesis': hypothesis,
            'rationale': rationale,
            'expected_gain': expected_gain,
            'expected_gain_derivation': expected_gain_derivation,
            'spec': spec,
            'code_diff': code_diff,
            # --- Guard 2 outcome, recorded before any metric is believed ---
            'drift_check': drift,
            # --- results ---
            'metrics': metrics,
            'diagnostic': diagnostic,
            'accepted': accepted,
            'best_so_far': best_so_far,
            'stall_count': stall_count,
            # --- Robustness axis ---
            'error': error,
            'recovery': recovery,
            # --- Feasibility axis ---
            'seconds': seconds,
            'tokens_in': tokens_in, 'tokens_out': tokens_out,
            'llm_usage': usage, 'proposer_model': (usage or {}).get('model', self.model),
            'api_recovery': api_recovery,
            'wall_clock_s': self._elapsed(),
            'cache': cache,
        }
        if extra:
            rec.update(extra)
        self._write(rec)
        return rec

    def summary(self, *, designated, designation_reason, converged_at,
                convergence_reason, final_metrics):
        rec = {
            'kind': 'RUN_SUMMARY', 'run_id': self.run_id, 'mode': self.mode,
            'iterations_used': self.iterations, 'iteration_cap': 50,
            'manual_interventions': self.interventions,
            'total_tokens_in': self.tokens_in, 'total_tokens_out': self.tokens_out,
            'total_tokens': self.tokens_in + self.tokens_out,
            'cache_read_input_tokens': self.cache_read,
            'cache_creation_input_tokens': self.cache_write,
            'proposer_model': self.model,
            'agent_wall_clock_s': self._elapsed(),
            'gpu_hours': 0.0,
            'designated_submission': designated,
            'designation_reason': designation_reason,
            'converged_at_iteration': converged_at,
            'convergence_reason': convergence_reason,
            'final_metrics': final_metrics,
            'git_commit': self._git_head(),
        }
        self._write(rec)
        return rec

    # -------- internals --------
    def _elapsed(self):
        return round(time.time() - self.t0, 1)

    @staticmethod
    def _git_head():
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    def _write(self, rec):
        with open(self.path, 'a') as fh:
            fh.write(json.dumps(rec, default=str) + '\n')
