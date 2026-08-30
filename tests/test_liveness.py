"""Liveness tests. Every path that consumes an iteration without running an
experiment must hit the abort well before the 50-iteration cap.

Motivated by a real failure: credit exhaustion made 50 iterations pass in 54
seconds, each component behaving correctly, the composition reporting a
converged run that had never proposed anything.
"""
import json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'agent')); sys.path.insert(0, os.path.join(ROOT, 'src'))
import controller as C
from proposer import ProposerError, Slate, Candidate, Spec, BlockChoice

PASS, FAIL = [], []
def check(n, cond, msg=''):
    (PASS if cond else FAIL).append(n)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{'' if cond else ': ' + msg}")

CAP = 50

def run_flat_fwd():
    def f(spec, seeds=None):
        return baseline_ok(spec, seeds) | {
            'metrics': {'primary': 0.6015, 'primary_std': 0.0002, 'GAUC': 0.66,
                        'nDCG@5': 0.53, 'n_seeds': 3}}
    return f


def build(backend, executor_run):
    """A Controller with the proposer/executor swapped out, so the loop is
    exercised without loading 1.4M rows or calling an API."""
    c = C.Controller.__new__(C.Controller)
    c.backend = backend
    c.proposer = type('P', (), {'propose': backend.propose})()
    c.ex = type('E', (), {'run': staticmethod(executor_run)})()
    path = tempfile.mktemp(suffix='.jsonl')
    from runlog import RunLog
    c.log = RunLog(path, 'liveness', mode='dev', model='test')
    c.max_iters, c.history, c.dead_actions = CAP, [], set()
    c.evaluated, c.repeat_rejections = {}, 0
    c.best, c.best_curve, c.stall = (0.6014, {'tag': 'BASE'}, 0), [0.6014], 0
    c.per_iter_converged_at, c._last_ran = None, False
    import time; c.t0 = time.time()
    c._path = path
    return c

def baseline_ok(spec, seeds=None):
    """Iteration 0 must reproduce the baseline or run() aborts before the loop."""
    return {'ok': True, 'rejected_by': None, 'drift': {'passed': True},
            'metrics': {'primary': 0.6014, 'primary_std': 0.0002, 'GAUC': 0.667,
                        'nDCG@5': 0.536, 'n_seeds': 3},
            'diagnostic': None, 'seconds': 1, 'cache': {}}


def spec_ok(tag='x'):
    return {'model': 'lightgbm', 'objective': 'binary', 'group_chunk': None,
            'feature_blocks': ['base5'], 'params': {}, 'seeds': [0], 'tag': tag}

print("=== mode 1: every proposal fails (the credit-exhaustion case) ===")
class AlwaysFails:
    last_recovery = []
    def propose(self, *a, **k): raise ProposerError('BadRequestError: 400 credit balance too low')
c = build(AlwaysFails(), baseline_ok)
summary = c.run()
n = summary['iterations_used']
check('aborts well before the cap', n < 10, f'used {n} of {CAP}')
check('abort names the liveness condition', 'liveness condition' in summary['convergence_reason'])
check('abort names the underlying error', 'credit balance' in summary['convergence_reason'])

print("\n=== mode 2: every slate is dead/invalid (no API failure at all) ===")
class InvalidSpecs:
    last_recovery = []
    def propose(self, *a, **k):
        return ([{'hypothesis': 'h', 'rationale': 'r', 'expected_gain': 0.01,
                  'expected_gain_derivation': '[POSITIVE_RATE]', 'tier': 'A',
                  'cited_facts': ['POSITIVE_RATE'],
                  'spec': {'model': 'NOT_A_MODEL', 'feature_blocks': []}}],
                {'input_tokens': 1, 'output_tokens': 1}, [])
c = build(InvalidSpecs(), baseline_ok)
summary = c.run()
n = summary['iterations_used']
check('aborts well before the cap', n < 10, f'used {n} of {CAP}')
check('abort cites the liveness condition', 'liveness condition' in summary['convergence_reason'])
check('reason explains convergence cannot fire on these',
      'Convergence cannot fire' in summary['convergence_reason'])

print("\n=== mode 3: every slate is an already-evaluated repeat ===")
SAME = spec_ok('same')
class AlwaysRepeats:
    last_recovery = []
    def propose(self, *a, **k):
        return ([{'hypothesis': 'h', 'rationale': 'r', 'expected_gain': 0.01,
                  'expected_gain_derivation': '[POSITIVE_RATE]', 'tier': 'A',
                  'cited_facts': ['POSITIVE_RATE'], 'spec': dict(SAME)}],
                {'input_tokens': 1, 'output_tokens': 1}, [])
calls = {'n': 0}
def run_once(spec, seeds=None):
    if spec.get('model') != 'fm':          # don't count the iteration-0 baseline
        calls['n'] += 1
    return baseline_ok(spec, seeds) | {
        'metrics': {'primary': 0.6015, 'primary_std': 0.0002, 'GAUC': 0.66,
                    'nDCG@5': 0.53, 'n_seeds': 3}}
c = build(AlwaysRepeats(), run_once)
summary = c.run()
n = summary['iterations_used']
check('aborts well before the cap', n < 12, f'used {n} of {CAP}')
check('the identical spec was executed only ONCE', calls['n'] == 1,
      f"executed {calls['n']}x (excluding the iteration-0 baseline)")
check('repeats were rejected by the cache', summary.get('repeat_specs_rejected', 0) > 0,
      str(summary.get('repeat_specs_rejected')))

print("\n=== a proposer defect that is NOT an API error must not kill the run ===")
# The scored-run crash: messages.parse() returned parsed_output=None for an
# incomplete response, so `slate.candidates` raised AttributeError -- a
# successful HTTP call with unusable content, caught by no API-error handler.
class RaisesNonApiError:
    last_recovery = []
    def propose(self, *a, **k):
        raise AttributeError("'NoneType' object has no attribute 'candidates'")
c = build(RaisesNonApiError(), baseline_ok)
summary = c.run()
check('run completes instead of crashing', isinstance(summary, dict))
check('stops via the liveness condition, not a traceback',
      'liveness condition' in summary['convergence_reason'], summary['convergence_reason'][:90])
check('the non-API error is recorded', 'NoneType' in summary['convergence_reason'],
      summary['convergence_reason'][:120])

print("\n=== an incomplete parse is retried, then degrades gracefully ===")
class IncompleteThenFine:
    """Mimics parsed_output=None: raises ProposerError after its retries."""
    last_recovery = [{'attempt': 1, 'error': 'IncompleteResponse', 'retryable': True}]
    def __init__(self): self.n = 0
    def propose(self, *a, **k):
        self.n += 1
        if self.n <= 2:
            from proposer import ProposerError
            raise ProposerError('proposer returned an incomplete response')
        return ([{'hypothesis': 'h', 'rationale': 'r', 'expected_gain': 0.01,
                  'expected_gain_derivation': '[POSITIVE_RATE]', 'tier': 'A',
                  'cited_facts': ['POSITIVE_RATE'],
                  'spec': dict(spec_ok(), params={'num_leaves': 31 + self.n})}],
                {'input_tokens': 1, 'output_tokens': 1}, [])
c = build(IncompleteThenFine(), run_flat_fwd())
summary = c.run()
check('two incomplete responses do not end the run',
      'liveness' not in summary['convergence_reason'], summary['convergence_reason'][:90])
check('the run goes on to converge normally',
      'window reading' in summary['convergence_reason'])

print("\n=== a productive iteration resets the counter ===")
state = {'i': 0}
class AlternatesFailure:
    last_recovery = []
    def propose(self, *a, **k):
        state['i'] += 1
        if state['i'] % 2 == 0:
            raise ProposerError('transient')
        return ([{'hypothesis': 'h', 'rationale': 'r', 'expected_gain': 0.01,
                  'expected_gain_derivation': '[POSITIVE_RATE]', 'tier': 'A',
                  'cited_facts': ['POSITIVE_RATE'],
                  # vary a field the canonical spec hash actually covers, so
                  # these are genuinely distinct experiments and not repeats
                  'spec': dict(spec_ok(), params={'num_leaves': 31 + state['i']})}],
                {'input_tokens': 1, 'output_tokens': 1}, [])
def run_flat(spec, seeds=None):
    return baseline_ok(spec, seeds) | {
        'metrics': {'primary': 0.6015, 'primary_std': 0.0002, 'GAUC': 0.66,
                    'nDCG@5': 0.53, 'n_seeds': 3}}
c = build(AlternatesFailure(), run_flat)
summary = c.run()
check('alternating failures do NOT trigger the liveness abort',
      'liveness' not in summary['convergence_reason'], summary['convergence_reason'][:90])
check('it converges normally instead', 'window reading' in summary['convergence_reason'])

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
