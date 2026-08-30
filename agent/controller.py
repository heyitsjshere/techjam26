"""Controller: gain-ranked scheduling, honest convergence, graceful recovery.

CONVERGENCE SEMANTICS
The organizers pin the CONSTANTS (epsilon 0.002, N 3, in baseline_scores.json)
but not the IMPLEMENTATION -- there is no convergence code anywhere in the
starter kit. The only prose is README:72-73, and it is ambiguous in the original
Chinese in exactly the way the English is: "improvement does not exceed 0.002
over 3 consecutive iterations" admits both

  window        : best(t) - best(t-N) <= eps          <- what we run
  per-iteration : each of the last N iterations gained <= eps

We run the window reading and compute the per-iteration reading in parallel,
logging the iteration at which each would fire, every run. See POLICY.md
section 9 for the ambiguity, the choice, and the reasoning. We disclose this
rather than quietly benefit from it.

CONVERGENCE HAZARD
Three flat iterations are easy to hit on a saturated model. Three mechanisms
address it, none of which hardcodes an ordering:
  1. Candidates execute in descending order of the proposer's OWN expected gain,
     which it derives from the briefing and must cite. This controller has no
     opinion about which move is structural.
  2. At stall == N-1 the controller forces a high-variance move.
  3. Improvement is measured against best-so-far over a 3-seed mean, so seed
     noise (std ~0.0008) cannot masquerade as progress or as saturation.
"""
import hashlib
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import actionspace
import briefing as briefing_mod
from executor import Executor, ExecutionError
from guards import GuardViolation
from proposer import Proposer, ProposerError
from runlog import RunLog

EPS = 0.002
N_STALL = 3
MAX_ITERS = 50
WALL_CLOCK_LIMIT_S = 6 * 3600
UNCITED_DISCOUNT = 0.5
MAX_RESLATES = 2          # bounded, so a stubborn proposer cannot spin forever

# An API failure must not kill the run -- but neither should it let the run spin
# through the whole iteration cap doing nothing. A dev run exhausted its credit
# balance and burned all 50 iterations in 54 seconds on consecutive 400s,
# reporting a converged run that had never proposed anything. After this many
# CONSECUTIVE proposal failures the run finishes with an explicit reason.
MAX_CONSECUTIVE_PROPOSAL_FAILURES = 3

# POLICY.md section 6: nothing single-seed is ever designated. Selection,
# convergence and designation all run on the 3-seed mean. Seed std is 0.0008 and
# real deltas here are the same order, so a single seed cannot separate signal
# from noise. ~3x wall-clock against a 6h ceiling on 20-60s fits is affordable,
# and Feasibility is graded in coarse tiers.
EVAL_SEEDS = [0, 1, 2]

# POLICY.md sections 6 and 11. A config is "structurally justified" when the
# iteration that produced it cited a fact naming a train/eval mismatch or a
# property of the metric's form -- as opposed to a parameter choice or a feature
# hunch. The band is the organizers' PUBLISHED 5-seed std, fixed in advance: a
# pre-registered constant cannot be tuned after seeing results, a data-dependent
# one could be.
STRUCTURAL_FACTS = frozenset({
    'GROUP_SHAPE_MISMATCH', 'POINTWISE_BASELINE',
    'USER_CONSTANT_NO_EFFECT', 'WITHIN_USER_RANKING'})
DESIGNATION_BAND = 0.0008
# Floating-point slack for the band comparison. A gap of exactly the band must
# designate structural (the rule says "within one seed std", inclusive), but
# 0.6030 - (0.6030 - 0.0008) evaluates to 0.0008000000000000229 in binary
# floating point and would fall the wrong side of a bare <=. This is decision
# logic for the submission, so the boundary is made deterministic rather than
# left to representation error.
DESIGNATION_BAND_TOL = 1e-9

# POLICY.md section 7, pre-registered.
BASELINE_TARGET = 0.6016
BASELINE_SEED_STD = 0.0008
BASELINE_TOLERANCE = 2 * BASELINE_SEED_STD


class Controller:
    def __init__(self, backend, log_path, run_id, mode='dev', max_iters=MAX_ITERS):
        if mode == 'scored' and backend.__class__.__name__ == 'StubBackend':
            raise RuntimeError('StubBackend is a plumbing test only; refused in a scored run')
        self.backend = backend
        self.proposer = Proposer(backend)
        self.ex = Executor()
        self.log = RunLog(log_path, run_id, mode=mode,
                          model=getattr(backend, 'model', 'stub'))
        self.max_iters = max_iters
        self.history = []
        self.dead_actions = set()
        self.evaluated = {}       # spec key -> prior result summary
        self.repeat_rejections = 0
        self.best = None            # (primary_mean, spec, iteration)
        self.best_curve = []        # best-so-far after each COUNTING iteration
        self.stall = 0              # per-iteration reading
        self.per_iter_converged_at = None
        self.t0 = time.time()

    # ---------- iteration 0 ----------
    def iteration_zero(self):
        """Task Requirement 1: the agent stands the pipeline up and reproduces
        the official baseline itself, logged like any other iteration.

        It seeds best-so-far and the convergence window, but is NOT an
        improvement attempt, so it does not start the stall counter -- the same
        principle applied elsewhere, that only iterations which ran an
        experiment are evidence about saturation.
        """
        spec = actionspace.default_spec()
        t0 = time.time()
        err = rec_err = None
        try:
            res = self.ex.run(spec, seeds=EVAL_SEEDS)
        except Exception as e:
            err, res = f'{type(e).__name__}: {e}', None
            rec_err = traceback.format_exc(limit=3)
        m = (res or {}).get('metrics')
        ok = bool(m) and abs(m['primary'] - BASELINE_TARGET) <= BASELINE_TOLERANCE
        if m:
            self.best = (m['primary'], spec, 0)
            self.best_curve.append(m['primary'])
        self.log.iteration(
            i=0, tier='A',
            hypothesis='Reproduce the official FM baseline through this pipeline '
                       'before proposing anything.',
            rationale='Every later delta is measured against this number. If the '
                      'pipeline cannot reproduce 0.6016 the harness is wrong and '
                      'no subsequent result means anything.',
            expected_gain=0.0,
            expected_gain_derivation='Not an improvement attempt; a correctness '
                                     'precondition. Target validation primary 0.6016.',
            spec=spec, drift=(res or {}).get('drift'), metrics=m,
            diagnostic=(res or {}).get('diagnostic'),
            accepted=ok, best_so_far=self.best[0] if self.best else None,
            stall_count=0, error=err, recovery=rec_err,
            seconds=round(time.time() - t0, 1), cache=(res or {}).get('cache'),
            extra={'baseline_reproduced': ok, 'baseline_target': BASELINE_TARGET,
                   'baseline_tolerance': BASELINE_TOLERANCE,
                   'baseline_deviation': None if not m else
                       round(m['primary'] - BASELINE_TARGET, 5),
                   'primary_std': (m or {}).get('primary_std'),
                   'n_seeds': (m or {}).get('n_seeds'),
                   'ran_experiment': False, 'counted_toward_convergence': False,
                   'note': 'seeds best-so-far and the convergence window; does '
                           'not start the stall counter'})
        self.history.append({'iteration': 0, 'spec': spec, 'metrics': m,
                             'primary': (m or {}).get('primary'),
                             'primary_std': (m or {}).get('primary_std'),
                             'cited_facts': [],
                             'note': 'baseline reproduction'})
        return ok

    # ---------- loop ----------
    def run(self):
        if not self.iteration_zero():
            return self._finish('baseline reproduction failed', None)
        i = 1
        reslates = 0
        proposal_failures = 0
        while i < self.max_iters:
            if time.time() - self.t0 > WALL_CLOCK_LIMIT_S:
                return self._finish('wall-clock backstop reached', i)
            forced = (self.stall == N_STALL - 1)
            try:
                cands, usage, api_recovery = self.proposer.propose(
                    briefing_mod.full_briefing(), self._action_doc(),
                    self.history[-12:], self.stall, forced)
            except ProposerError as e:
                self.log.iteration(
                    i=i, tier='A', hypothesis='(proposal failed)', rationale='',
                    expected_gain=0.0, expected_gain_derivation='', spec=None,
                    error=str(e),
                    recovery='proposal abandoned; requesting a fresh slate next '
                             'iteration. Run continues.',
                    api_recovery=list(getattr(self.backend, 'last_recovery', [])),
                    drift=None, metrics=None, diagnostic=None, accepted=False,
                    best_so_far=self.best[0] if self.best else None,
                    stall_count=self.stall, seconds=0,
                    extra={'ran_experiment': False,
                           'counted_toward_convergence': False,
                           'consecutive_proposal_failures': proposal_failures + 1})
                proposal_failures += 1
                if proposal_failures >= MAX_CONSECUTIVE_PROPOSAL_FAILURES:
                    return self._finish(
                        f'aborted after {proposal_failures} consecutive proposal '
                        f'failures; the proposer could not be reached, so no '
                        f'further experiment was possible. Last error: {e}', i)
                i += 1
                continue

            proposal_failures = 0          # a successful slate resets the counter
            outcome = self._execute_first_viable(
                self._rank(cands), i, usage, api_recovery, forced)
            if outcome == 'all_repeats' and reslates < MAX_RESLATES:
                # Every candidate was already evaluated. Re-running one is
                # deterministic given fixed seeds, so it cannot produce new
                # information. Ask for a fresh slate WITHOUT spending the
                # iteration -- run 1 of the previous round burned its forced
                # high-variance iteration on a byte-identical repeat.
                reslates += 1
                continue
            reslates = 0
            if self._converged_window():
                return self._finish(
                    f'window reading: best-so-far improved by no more than {EPS} '
                    f'across the last {N_STALL} counting iterations', i)
            i += 1
        return self._finish(f'iteration cap {self.max_iters} reached', i)

    def _rank(self, cands):
        for c in cands:
            g = float(c.get('expected_gain', 0.0))
            c['_effective_gain'] = g * (1.0 if c.get('cited_facts') else UNCITED_DISCOUNT)
            c['_uncited'] = not c.get('cited_facts')
        return sorted(cands, key=lambda c: -c['_effective_gain'])

    def _execute_first_viable(self, ranked, i, usage, api_recovery, forced):
        repeats = []
        for c in ranked:
            key = self._key(c.get('spec'))
            if key in self.dead_actions:
                continue
            if key in self.evaluated:
                repeats.append({'hypothesis': c.get('hypothesis'),
                                'spec': c.get('spec'),
                                'already_evaluated': self.evaluated[key]})
                continue
            errs = actionspace.validate(c.get('spec') or {})
            attempts, err, rec, res = 0, None, None, None
            while attempts < 2:
                if errs:
                    err, rec = f'invalid spec: {errs}', 'schema rejected; next candidate'
                    break
                try:
                    res = self.ex.run(c['spec'], seeds=EVAL_SEEDS)
                    err = rec = None
                    break
                except GuardViolation:
                    raise
                except Exception as e:
                    attempts += 1
                    err = f'{type(e).__name__}: {e}'
                    rec = (f'retry {attempts}/2 with error fed back' if attempts < 2
                           else 'two failures: action marked dead, routing around it')
            if err and res is None:
                self.dead_actions.add(key)
                self._log_iter(i, c, None, err, rec, usage, api_recovery, forced)
                return None
            if res is not None and (res.get('metrics') or res.get('rejected_by')):
                self.evaluated[key] = {
                    'iteration': i,
                    'primary': (res.get('metrics') or {}).get('primary'),
                    'primary_std': (res.get('metrics') or {}).get('primary_std'),
                    'rejected_by': res.get('rejected_by')}
            self._log_iter(i, c, res, None, None, usage, api_recovery, forced)
            return 'ran'
        if repeats:
            # Feed the prior results back so the next slate is informed, and log
            # the rejection so the cache firing is visible in the run record.
            self.repeat_rejections += len(repeats)
            for rp in repeats:
                self.history.append({
                    'iteration': i, 'hypothesis': rp['hypothesis'],
                    'spec': rp['spec'], 'rejected_by': 'already_evaluated',
                    'primary': rp['already_evaluated']['primary'],
                    'note': ('This exact spec was already evaluated at iteration '
                             f"{rp['already_evaluated']['iteration']} scoring "
                             f"{rp['already_evaluated']['primary']}. Re-running it is "
                             'deterministic and cannot produce new information. '
                             'Propose something different.')})
            self.log.iteration(
                i=i, tier='A',
                hypothesis='(slate contained only already-evaluated specs)',
                rationale='', expected_gain=0.0, expected_gain_derivation='',
                spec=None, drift=None, metrics=None, diagnostic=None,
                accepted=False, best_so_far=self.best[0] if self.best else None,
                stall_count=self.stall,
                error=None,
                recovery=f'{len(repeats)} repeat spec(s) rejected from the cache; '
                         'prior results fed back and a fresh slate requested '
                         'WITHOUT consuming the iteration',
                usage=usage, api_recovery=api_recovery, seconds=0,
                extra={'rejected_by': 'already_evaluated',
                       'repeats': repeats,
                       'repeat_rejections_total': self.repeat_rejections,
                       'ran_experiment': False,
                       'counted_toward_convergence': False})
            return 'all_repeats'
        self._log_iter(i, {'hypothesis': '(all candidates dead or invalid)',
                           'rationale': '', 'expected_gain': 0.0,
                           'expected_gain_derivation': '', 'tier': 'A', 'spec': None},
                       None, 'no viable candidate', 'requesting a fresh slate',
                       usage, api_recovery, forced)
        return None

    def _log_iter(self, i, c, res, err, rec, usage, api_recovery, forced):
        m = (res or {}).get('metrics')
        rejected = (res or {}).get('rejected_by')
        prev_best = self.best[0] if self.best else None
        ran = (m is not None) or (rejected == 'drift_check')

        improved = m is not None and (prev_best is None or m['primary'] > prev_best)
        if improved:
            self.best = (m['primary'], c['spec'], i)
        if ran:
            self.best_curve.append(self.best[0] if self.best else 0.0)
            gained = (m is not None and
                      (prev_best is None or m['primary'] > prev_best + EPS))
            self.stall = 0 if gained else self.stall + 1
            if self.stall >= N_STALL and self.per_iter_converged_at is None:
                self.per_iter_converged_at = i

        self.log.iteration(
            i=i, tier=c.get('tier', 'A'), hypothesis=c.get('hypothesis'),
            rationale=c.get('rationale'), expected_gain=c.get('expected_gain'),
            expected_gain_derivation=c.get('expected_gain_derivation'),
            spec=c.get('spec'), code_diff=c.get('code_diff'),
            drift=(res or {}).get('drift'), metrics=m,
            diagnostic=(res or {}).get('diagnostic'),
            accepted=improved, best_so_far=self.best[0] if self.best else None,
            stall_count=self.stall, error=err, recovery=rec,
            seconds=(res or {}).get('seconds'),
            tokens_in=(usage or {}).get('input_tokens', 0),
            tokens_out=(usage or {}).get('output_tokens', 0),
            usage=usage, api_recovery=api_recovery, cache=(res or {}).get('cache'),
            extra={'rejected_by': rejected, 'forced_high_variance': forced,
                   'cited_facts': c.get('cited_facts'),
                   'block_justifications': (c.get('spec') or {}).get('block_justifications'),
                   'expected_gain_effective': c.get('_effective_gain'),
                   'uncited_discount_applied': c.get('_uncited'),
                   'ran_experiment': ran, 'counted_toward_convergence': ran,
                   'primary_std': (m or {}).get('primary_std'),
                   'n_seeds': (m or {}).get('n_seeds'),
                   'convergence_window_delta': self._window_delta(),
                   'per_iteration_would_have_converged_at': self.per_iter_converged_at})
        self.history.append({
            'iteration': i, 'hypothesis': c.get('hypothesis'), 'spec': c.get('spec'),
            'expected_gain': c.get('expected_gain'),
            'primary': (m or {}).get('primary'),
            'primary_std': (m or {}).get('primary_std'),
            'diagnostic': (res or {}).get('diagnostic'),
            'rejected_by': rejected, 'error': err,
            # read by _designate(); POLICY.md sections 6 and 11
            'cited_facts': c.get('cited_facts') or [],
            'expected_gain_derivation': c.get('expected_gain_derivation')})

    def _window_delta(self):
        """best(t) - best(t-N) over counting iterations. None until N+1 exist."""
        if len(self.best_curve) < N_STALL + 1:
            return None
        return round(self.best_curve[-1] - self.best_curve[-1 - N_STALL], 6)

    def _converged_window(self):
        d = self._window_delta()
        return d is not None and d <= EPS

    def _finish(self, reason, i):
        best_primary = self.best[0] if self.best else None
        designated, why = self._designate()
        return self.log.summary(
            designated=designated, designation_reason=why, converged_at=i,
            convergence_reason=reason,
            final_metrics={'valid_primary': best_primary,
                           'delta_vs_fm': None if best_primary is None
                           else round(best_primary - BASELINE_TARGET, 5)},
            extra={'convergence_reading_used': 'window',
                   'converged_at_window': i,
                   'converged_at_per_iteration': self.per_iter_converged_at,
                   'best_curve': [round(x, 5) for x in self.best_curve],
                   'eval_seeds': EVAL_SEEDS,
                   'repeat_specs_rejected': self.repeat_rejections,
                   'distinct_specs_evaluated': len(self.evaluated)})

    def _designate(self):
        """Execute the pre-registered designation rule. POLICY.md sections 6, 11, 13.

        This runs the rule; it does not describe it and leave it to a human.
        Every branch is logged, and the one case the rule cannot resolve calls
        record_intervention(), so a run that needed a human to choose reports a
        non-zero intervention count rather than a silent one.

        Iteration 0 is NOT a designation candidate. It reproduces the baseline as
        a correctness precondition -- the same reason it does not start the stall
        counter (section 10). Allowing it into the candidate pool let a dev run
        designate a config that scored BELOW the baseline: with no experiment
        beating iteration 0, best-valid became the FM baseline itself and the
        within-band rule then handed designation to a worse structural config.
        """
        baseline = next((h.get('primary') for h in self.history
                         if h.get('iteration') == 0), None)

        # Candidates are EXPERIMENTS only. Iteration 0 is excluded.
        experiments = [h for h in self.history
                       if h.get('iteration', 0) >= 1 and h.get('primary') is not None]

        if not experiments:
            self.log.record_intervention(
                'designation rule could not resolve: no experiment iteration '
                'produced a metric, so there is no candidate submission at all. '
                'A human must decide what, if anything, to submit.', who='rule')
            return None, ('No experiment iteration produced a metric. Iteration 0 '
                          'is a correctness precondition and is not a designation '
                          'candidate, so there is nothing to designate. An '
                          'intervention has been RECORDED.')

        bv = max(experiments, key=lambda h: h['primary'])
        bv_primary, bv_iter, bv_spec = bv['primary'], bv['iteration'], bv['spec']
        structural = [h for h in experiments
                      if set(h.get('cited_facts') or []) & STRUCTURAL_FACTS]

        def rec(branch, spec, primary, reason, st=None):
            beats = baseline is None or primary > baseline
            if not beats:
                reason += (f' WARNING: the designated config ({primary:.5f}) does '
                           f'NOT beat the iteration-0 baseline ({baseline:.5f}). '
                           f'This run produced no improvement and the submission '
                           f'is reported as such.')
            self.log.iteration(
                i=-1, tier='A',
                hypothesis='DESIGNATION: apply the pre-registered rule '
                           '(POLICY.md sections 6, 11, 13)',
                rationale=reason, expected_gain=0.0,
                expected_gain_derivation='Not an experiment; the designation rule.',
                spec=spec, drift=None, metrics=None, diagnostic=None,
                accepted=True, best_so_far=bv_primary, stall_count=self.stall,
                seconds=0,
                extra={'kind_detail': 'DESIGNATION_RECORD', 'branch': branch,
                       'band': DESIGNATION_BAND,
                       'structural_fact_keys': sorted(STRUCTURAL_FACTS),
                       'baseline_primary': baseline,
                       'designated_primary': primary,
                       'beats_baseline': beats,
                       'candidate_pool_excludes_iteration_0': True,
                       'best_valid': {'iteration': bv_iter, 'primary': bv_primary,
                                      'primary_std': bv.get('primary_std'),
                                      'cited_facts': bv.get('cited_facts')},
                       'structural': None if st is None else {
                           'iteration': st.get('iteration'), 'primary': st.get('primary'),
                           'primary_std': st.get('primary_std'),
                           'cited_facts': st.get('cited_facts')},
                       'ran_experiment': False,
                       'counted_toward_convergence': False})
            return spec, reason

        if not structural:
            self.log.record_intervention(
                'designation rule could not resolve: no experiment iteration '
                'cited a structural fact key, so no structurally-justified '
                'candidate exists to compare against best-valid. A human must '
                'choose.', who='rule')
            return rec('NO_STRUCTURAL_CANDIDATE', bv_spec, bv_primary,
                       f'No experiment cited any of {sorted(STRUCTURAL_FACTS)}, so '
                       f'the §6 comparison could not be made. Falling back to '
                       f'best-valid: iteration {bv_iter}, 3-seed mean '
                       f'{bv_primary:.5f} (std {bv.get("primary_std")}). An '
                       f'intervention has been RECORDED because this decision was '
                       f'not resolved by the rule.')

        st = max(structural, key=lambda h: h['primary'])
        gap = bv_primary - st['primary']
        std_note = (f'best-valid std {bv.get("primary_std")}, structural std '
                    f'{st.get("primary_std")}, band {DESIGNATION_BAND} '
                    f'(published 5-seed std)')

        if st['iteration'] == bv_iter:
            return rec('NO_DIVERGENCE', bv_spec, bv_primary,
                       f'No divergence: the best-valid config IS the '
                       f'structurally-justified one (iteration {bv_iter}, 3-seed '
                       f'mean {bv_primary:.5f}, cited {st.get("cited_facts")}). The '
                       f'rule was executed and the two criteria agreed. {std_note}.',
                       st)

        if gap <= DESIGNATION_BAND + DESIGNATION_BAND_TOL:
            return rec('STRUCTURAL_WITHIN_BAND', st['spec'], st['primary'],
                       f'DIVERGED, structural designated. Best-valid: iteration '
                       f'{bv_iter}, 3-seed mean {bv_primary:.5f}. Structural: '
                       f'iteration {st["iteration"]}, 3-seed mean {st["primary"]:.5f}, '
                       f'cited {st.get("cited_facts")}. Gap {gap:.5f} is within the '
                       f'band, so §6 designates the structurally-justified config: '
                       f'79 valid-selected experiments carry selection risk that a '
                       f'structural fix does not. {std_note}.', st)

        return rec('BEST_VALID_BEYOND_BAND', bv_spec, bv_primary,
                   f'DIVERGED beyond the band, best-valid designated. Best-valid: '
                   f'iteration {bv_iter}, 3-seed mean {bv_primary:.5f}. Best '
                   f'structural: iteration {st["iteration"]}, 3-seed mean '
                   f'{st["primary"]:.5f}, cited {st.get("cited_facts")}. Gap '
                   f'{gap:.5f} exceeds the band, so §6 as clarified in §11 '
                   f'designates best-valid and logs the divergence. {std_note}.', st)

    @staticmethod
    def _key(spec):
        """Canonical hash of everything that determines the result.

        Seeds are excluded because the controller forces EVAL_SEEDS, and
        block_justifications are excluded because they are prose about the spec,
        not part of it. Params ARE included: two specs differing only in a
        hyperparameter are different experiments, even if we expect the axis to
        be flat -- the cache must not silently swallow a real question.
        """
        if not spec:
            return 'none'
        payload = {
            'model': spec.get('model'), 'objective': spec.get('objective'),
            'group_chunk': spec.get('group_chunk'),
            'feature_blocks': sorted(spec.get('feature_blocks') or []),
            'params': dict(sorted((spec.get('params') or {}).items())),
            'recency_decay': spec.get('recency_decay'),
            'min_date': spec.get('min_date'),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]

    @staticmethod
    def _action_doc():
        return (f"MODELS: {actionspace.MODELS}\nOBJECTIVES: {actionspace.OBJECTIVES}\n"
                f"GROUP_CHUNKS: {actionspace.GROUP_CHUNKS} -- {actionspace.GROUP_CHUNK_DOC}\n"
                f"FEATURE_BLOCKS: {actionspace.FEATURE_BLOCKS}\n"
                f"PARAM_GRID: {actionspace.PARAM_GRID}\n"
                f"TRAIN_SHAPING: {actionspace.TRAIN_SHAPING}\n"
                f"ENSEMBLE: {actionspace.ENSEMBLE}\nFORBIDDEN: {actionspace.FORBIDDEN}")
