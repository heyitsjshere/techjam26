"""Guard tests, including empirical calibration of the drift threshold against
Phase 1's known-good and known-leaking features."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'agent'))
import numpy as np
import guards, loader, features as F

S = loader.load_agent(); store = F.FeatureStore().fit(S['train'])
PASS, FAIL = [], []
def check(n, cond, msg=''):
    (PASS if cond else FAIL).append(n)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{'' if cond else ': '+msg}")

print("=== Guard 1: out-of-fold only ===")
try:
    with guards.agent_mode(): store.build(S['train'], ('item_agg',), encoding='loo')
    check('loo raises in agent mode', False, 'no exception')
except guards.GuardViolation:
    check('loo raises in agent mode', True)
try:
    with guards.agent_mode(): store.build(S['train'], ('item_agg',), encoding='naive')
    check('naive raises in agent mode', False, 'no exception')
except guards.GuardViolation:
    check('naive raises in agent mode', True)
with guards.agent_mode():
    check('oof works in agent mode', store.build(S['train'], ('item_agg',), encoding='oof').shape[0] == len(S['train']))
    # Regression: the guard must NOT fire on the eval path. Applying a
    # train-fitted statistic to valid rows is the ordinary causal join -- those
    # rows are in a disjoint later window and contributed nothing to it, so
    # there is no own-row to exclude. Caught by the first full-loop plumbing run.
    try:
        n = store.build(S['valid'], ('item_agg', 'user_agg')).shape[0]
        check('eval path is NOT blocked by Guard 1', n == len(S['valid']))
    except guards.GuardViolation as e:
        check('eval path is NOT blocked by Guard 1', False, f'guard over-fired: {e}')
    # Stronger than "eval rejects a bad mode": build() forces 'apply' on the
    # eval path, so the encoding argument is inert there and a caller cannot
    # select a leaky scheme for eval rows at all.
    import numpy as _np
    a = store.build(S['valid'], ('item_agg',), encoding='oof')
    b = store.build(S['valid'], ('item_agg',), encoding='naive')
    c = store.build(S['valid'], ('item_agg',), encoding='loo')
    check('eval path ignores the encoding argument entirely',
          _np.array_equal(a.to_numpy(), b.to_numpy()) and
          _np.array_equal(a.to_numpy(), c.to_numpy()),
          'eval output depended on the encoding argument')
check('spec schema has no encoding field',
      bool(__import__('actionspace').validate(
          {'model':'lightgbm','objective':'binary','group_chunk':None,
           'feature_blocks':['base5'],'encoding':'loo'})))

print("\n=== Guard 2: drift threshold calibration ===")
va = S['valid']
cases = []
for blk, enc, loo, label in [
    (('item_agg',), 'oof', None, 'item_agg OOF (legitimate)'),
    (('dur_feats',), 'oof', None, 'dur_feats (legitimate)'),
    (('duration',), 'oof', None, 'duration (legitimate)'),
    (('cross_agg',), 'oof', None, 'cross_agg LOO-corrected (legitimate)'),
    (('cross_agg',), 'oof', False, 'cross_agg NAIVE  (KNOWN LEAK)'),
]:
    kw = {} if loo is None else {'loo': loo}
    tr = store.build(S['train'], blk, encoding=enc, **kw)
    vv = store.build(va, blk, encoding=enc)
    rep = guards.drift_report(tr, vv)
    mx = max((r['smd'] for r in rep if not r['exempt']), default=0.0)
    worst = max((r for r in rep if not r['exempt']), key=lambda r: r['smd'], default=None)
    cases.append((label, mx, worst))
    print(f"  {label:<38s} max SMD {mx:7.4f}   worst: {worst['feature'] if worst else '-'}")

legit = [m for l, m, _ in cases if 'KNOWN LEAK' not in l]
leak  = [m for l, m, _ in cases if 'KNOWN LEAK' in l]
print(f"\n  legitimate blocks max SMD : {max(legit):.4f}")
print(f"  known-leaking block SMD   : {max(leak):.4f}")
print(f"  threshold in use          : {guards.DRIFT_SMD_THRESHOLD}")
check("threshold accepts every legitimate block", max(legit) < guards.DRIFT_SMD_THRESHOLD,
      f'legit max {max(legit):.4f} >= {guards.DRIFT_SMD_THRESHOLD}')
check('threshold rejects the known leak', max(leak) > guards.DRIFT_SMD_THRESHOLD,
      f'leak {max(leak):.4f} <= {guards.DRIFT_SMD_THRESHOLD}')
check('separation is at least 2x', max(leak) > 2 * max(legit),
      f'{max(leak):.4f} vs 2*{max(legit):.4f}')

print("\n=== Guard 2 wiring: a drift failure blocks the metric ===")
p, rep = guards.check_drift(store.build(S['train'], ('cross_agg',), loo=False),
                            store.build(va, ('cross_agg',)), where='test')
check('leaking block fails check_drift', not p)
check('failure names the offending columns', bool(rep['failed']), str(rep)[:120])
print(f"  failed columns: {rep['failed']}  max_smd={rep['max_smd']}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
