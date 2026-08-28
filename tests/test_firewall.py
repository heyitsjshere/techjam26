"""Firewall + loader-equivalence tests. Run before every Phase 1 experiment."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))

import numpy as np
import loader, metrics
from firewall import FirewallBreach, VALID_END, TEST_START

PASS, FAIL = [], []
def check(name, fn):
    try:
        fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e)); print(f"  FAIL  {name}: {type(e).__name__}: {e}")

print("=== Lock 1: agent loader cannot produce test rows ===")
S = loader.load_agent(use_cache=False)
check("load_agent returns exactly {train, valid}",
      lambda: (_ for _ in ()).throw(AssertionError(set(S))) if set(S) != {'train','valid'} else None)
check("no agent row is dated past VALID_END",
      lambda: [None for s in S.values() if s.max_date > VALID_END] and
              (_ for _ in ()).throw(AssertionError('test row present')) or None)
check("feature frames carry no denied same-row column",
      lambda: [__import__('firewall').assert_no_deny_columns(s.X.columns, 'test') for s in S.values()])
check("label y is 0/1 only",
      lambda: [None if set(np.unique(s.y)) <= {0,1} else (_ for _ in ()).throw(AssertionError(s.name)) for s in S.values()])

print("\n=== Lock 2: evaluate wrapper rejects test-window data ===")
def breach_by_date():
    s = S['valid']
    forged = loader.Split('valid', s.X, s.y, s.users,
                          np.full(len(s), TEST_START, np.int32))
    metrics.score(forged, np.zeros(len(s)))
def breach_by_name():
    s = S['valid']
    forged = loader.Split('test', s.X, s.y, s.users, s.dates)
    metrics.score(forged, np.zeros(len(s)))
for nm, fn in (("test-window dates raise FirewallBreach", breach_by_date),
               ("split named 'test' raises FirewallBreach", breach_by_name)):
    try:
        fn(); FAIL.append((nm, 'NO BREACH RAISED')); print(f"  FAIL  {nm}: firewall did not fire")
    except FirewallBreach:
        PASS.append(nm); print(f"  PASS  {nm}")
    except Exception as e:
        FAIL.append((nm, e)); print(f"  FAIL  {nm}: wrong exception {type(e).__name__}")

print("\n=== agent module graph never reaches the human-only test path ===")
def no_import():
    import ast, pathlib
    bad = []
    for p in pathlib.Path(os.path.join(ROOT,'src')).glob('*.py'):
        if p.name == 'human_only_test_scoring.py': continue
        for n in ast.walk(ast.parse(p.read_text())):
            names = ([a.name for a in n.names] if isinstance(n, ast.Import)
                     else [n.module or ''] if isinstance(n, ast.ImportFrom) else [])
            if any('human_only' in x for x in names): bad.append(p.name)
    if bad: raise AssertionError(f"agent modules import the human-only path: {bad}")
check("no src/ module imports human_only_test_scoring", no_import)

print("\n=== row order is bit-identical to organizers' data.load() ===")
def order_matches():
    import data as official
    off = official.load(loader.DATA_DIR)
    for name in ('train', 'valid'):
        ours, ref = S[name], off[name]
        assert len(ours) == len(ref), f"{name}: {len(ours)} vs {len(ref)}"
        u_ours = ours.X['user_id'].to_numpy()
        v_ours = ours.X['video_id'].to_numpy()
        u_ref = np.array([int(x[1]) for x in ref], np.int32)
        v_ref = np.array([int(x[2]) for x in ref], np.int32)
        y_ref = np.array([x[6] for x in ref], np.int8)
        assert np.array_equal(u_ours, u_ref), f"{name}: user_id order differs"
        assert np.array_equal(v_ours, v_ref), f"{name}: video_id order differs"
        assert np.array_equal(ours.y, y_ref), f"{name}: labels differ"
check("train+valid row order and labels match reference loader", order_matches)

print("\n=== train_outcomes is clamped to the train window ===")
def outcomes_clamped():
    df = loader.train_outcomes(use_cache=False)
    assert int(df['date'].max()) <= 20220421, df['date'].max()
    assert 'is_click' in df.columns and 'long_view' in df.columns
check("train_outcomes max date <= 20220421 and exposes outcome cols", outcomes_clamped)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
