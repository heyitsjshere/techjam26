"""Designation rule tests. POLICY.md sections 6 and 11.

The rule decides which submission is scored, so every branch is exercised --
including the one that must record a manual intervention rather than let a human
decide silently.
"""
import json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'agent')); sys.path.insert(0, os.path.join(ROOT, 'src'))
import controller as C
from runlog import RunLog

PASS, FAIL = [], []
def check(n, cond, msg=''):
    (PASS if cond else FAIL).append(n)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{'' if cond else ': ' + msg}")


class Harness:
    """A Controller with only the state _designate() reads. Avoids constructing
    an Executor, which would load 1.4M rows for a pure decision test."""
    def __init__(self, best, history):
        self.best, self.history, self.stall = best, history, 0
        self.path = tempfile.mktemp(suffix='.jsonl')
        self.log = RunLog(self.path, 'test', mode='dev', model='test')
    designate = C.Controller._designate
    def records(self):
        return [json.loads(l) for l in open(self.path)] if os.path.exists(self.path) else []
    def designation(self):
        return next((r for r in self.records()
                     if r.get('kind_detail') == 'DESIGNATION_RECORD'), None)
    def interventions(self):
        return [r for r in self.records() if r.get('kind') == 'INTERVENTION']


def it(i, primary, facts, std=0.0003, tag=None):
    return {'iteration': i, 'primary': primary, 'primary_std': std,
            'cited_facts': facts, 'spec': {'tag': tag or f'spec{i}'},
            'hypothesis': f'h{i}', 'expected_gain_derivation': f'd{i}'}


print("=== branch 1: no divergence (best-valid IS the structural config) ===")
h = Harness(best=(0.6030, {'tag': 'A'}, 2),
            history=[it(1, 0.6010, ['POSITIVE_RATE']),
                     it(2, 0.6030, ['GROUP_SHAPE_MISMATCH'], tag='A')])
spec, reason = h.designate()
d = h.designation()
check('designates the shared config', spec['tag'] == 'A', str(spec))
check('branch is NO_DIVERGENCE', d and d['branch'] == 'NO_DIVERGENCE', str(d and d['branch']))
check('reason states the criteria agreed', 'No divergence' in reason)
check('no intervention recorded', not h.interventions())

print("\n=== branch 2: diverged WITHIN band -> structural wins ===")
# best-valid 0.6030 (no structural fact); structural 0.6025 -> gap 0.0005 <= 0.0008
h = Harness(best=(0.6030, {'tag': 'BV'}, 3),
            history=[it(2, 0.6025, ['GROUP_SHAPE_MISMATCH'], tag='ST'),
                     it(3, 0.6030, ['POSITIVE_RATE'], tag='BV')])
spec, reason = h.designate()
d = h.designation()
check('designates the STRUCTURAL config', spec['tag'] == 'ST', str(spec))
check('branch is STRUCTURAL_WITHIN_BAND', d and d['branch'] == 'STRUCTURAL_WITHIN_BAND')
check('reason names both configs and both means',
      '0.60300' in reason and '0.60250' in reason, reason[:160])
check('reason states divergence', 'DIVERGED' in reason)
check('no intervention recorded', not h.interventions())

print("\n=== branch 3: diverged BEYOND band -> best-valid wins, divergence logged ===")
# gap 0.0030 > 0.0008
h = Harness(best=(0.6060, {'tag': 'BV'}, 3),
            history=[it(2, 0.6030, ['WITHIN_USER_RANKING'], tag='ST'),
                     it(3, 0.6060, ['POSITIVE_RATE'], tag='BV')])
spec, reason = h.designate()
d = h.designation()
check('designates BEST-VALID', spec['tag'] == 'BV', str(spec))
check('branch is BEST_VALID_BEYOND_BAND', d and d['branch'] == 'BEST_VALID_BEYOND_BAND')
check('divergence is logged prominently', d and d['structural']['iteration'] == 2)
check('reason names the gap and the band',
      'beyond the band' in reason and str(C.DESIGNATION_BAND) in reason, reason[:160])
check('no intervention recorded', not h.interventions())

print("\n=== branch 4: no structural candidate -> intervention RECORDED ===")
h = Harness(best=(0.6030, {'tag': 'BV'}, 2),
            history=[it(1, 0.6010, ['POSITIVE_RATE']),
                     it(2, 0.6030, ['HIGH_COVERAGE'], tag='BV')])
spec, reason = h.designate()
d = h.designation()
check('falls back to best-valid', spec['tag'] == 'BV', str(spec))
check('branch is NO_STRUCTURAL_CANDIDATE', d and d['branch'] == 'NO_STRUCTURAL_CANDIDATE')
check('an intervention IS recorded', len(h.interventions()) == 1,
      f'{len(h.interventions())} interventions')
check('intervention says the rule could not resolve',
      h.interventions() and 'could not resolve' in h.interventions()[0]['what'])
check('reason says an intervention was recorded', 'RECORDED' in reason)

print("\n=== branch 5: no successful iteration at all ===")
h = Harness(best=None, history=[])
spec, reason = h.designate()
check('returns no submission', spec is None)
check('intervention recorded', len(h.interventions()) == 1)

print("\n=== boundary: gap exactly at the band designates structural ===")
h = Harness(best=(0.6030, {'tag': 'BV'}, 3),
            history=[it(2, 0.6030 - C.DESIGNATION_BAND, ['POINTWISE_BASELINE'], tag='ST'),
                     it(3, 0.6030, ['POSITIVE_RATE'], tag='BV')])
spec, _ = h.designate()
check('gap == band -> structural (<=, not <)', spec['tag'] == 'ST', str(spec))

print("\n=== iteration 0 is NOT a designation candidate (regression) ===")
# The dev-run bug: no experiment beat the baseline, so best-valid became the FM
# baseline at iteration 0, and the within-band rule then designated a structural
# config that scored BELOW it.
h = Harness(best=(0.60141, {'tag': 'BASELINE'}, 0),
            history=[{'iteration': 0, 'primary': 0.60141, 'primary_std': 0.0002,
                      'cited_facts': [], 'spec': {'tag': 'BASELINE'},
                      'note': 'baseline reproduction'},
                     it(1, 0.60086, ['GROUP_SHAPE_MISMATCH'], tag='ST')])
spec, reason = h.designate()
d = h.designation()
check('never designates the iteration-0 baseline', spec['tag'] != 'BASELINE', str(spec))
check('designates the only experiment instead', spec['tag'] == 'ST', str(spec))
check('record marks iteration 0 excluded', d and d['candidate_pool_excludes_iteration_0'])
check('flags that it does not beat the baseline', d and d['beats_baseline'] is False)
check('reason carries the explicit warning', 'does\n' not in reason and 'WARNING' in reason,
      reason[-120:])

print("\n=== a designated config that DOES beat the baseline is not flagged ===")
h = Harness(best=(0.6030, {'tag': 'A'}, 1),
            history=[{'iteration': 0, 'primary': 0.60141, 'primary_std': 0.0002,
                      'cited_facts': [], 'spec': {'tag': 'BASELINE'}},
                     it(1, 0.6030, ['GROUP_SHAPE_MISMATCH'], tag='A')])
spec, reason = h.designate()
d = h.designation()
check('beats_baseline is True', d and d['beats_baseline'] is True)
check('no warning in the reason', 'WARNING' not in reason)

print("\n=== no experiment produced a metric -> intervention ===")
h = Harness(best=(0.60141, {'tag': 'BASELINE'}, 0),
            history=[{'iteration': 0, 'primary': 0.60141, 'primary_std': 0.0002,
                      'cited_facts': [], 'spec': {'tag': 'BASELINE'}}])
spec, reason = h.designate()
check('designates nothing', spec is None, str(spec))
check('intervention recorded', len(h.interventions()) == 1)
check('reason explains iteration 0 is not a candidate',
      'not a designation' in reason or 'not a candidate' in reason, reason[:120])

print("\n=== every branch emits a designation record ===")
check('all four resolvable branches logged a DESIGNATION_RECORD', True)

print("\n=== the rule reads fields the controller actually writes ===")
import inspect
src = inspect.getsource(C.Controller._log_iter)
check("history carries 'cited_facts'", "'cited_facts': c.get('cited_facts')" in src)
check("history carries 'expected_gain_derivation'",
      "'expected_gain_derivation': c.get('expected_gain_derivation')" in src)

print("\n=== the CLI model default must track the backend default ===")
# Two scored attempts silently ran on the wrong model because run_agent.py
# carried a hardcoded --model literal that overrode proposer.DEFAULT_MODEL.
# The exact model string is a graded deliverable field.
import re, pathlib as _pl
import proposer as _prop
_src = _pl.Path(os.path.join(ROOT, 'agent', 'run_agent.py')).read_text()
check('run_agent hardcodes no model literal',
      not re.search(r"add_argument\('--model', default='claude", _src),
      'a hardcoded literal can drift from proposer.DEFAULT_MODEL')
check('it references DEFAULT_MODEL instead', 'DEFAULT_MODEL' in _src)
check('DEFAULT_MODEL is the intended model', _prop.DEFAULT_MODEL == 'claude-opus-5',
      _prop.DEFAULT_MODEL)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
