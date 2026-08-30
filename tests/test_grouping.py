"""Regression tests for ranking-group construction.

The case that broke production: a user with fewer rows than the chunk size.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from grouping import group_sizes

PASS, FAIL = [], []
def check(n, cond, msg=''):
    (PASS if cond else FAIL).append(n)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{'' if cond else ': ' + msg}")

print("=== the case that broke the executor ===")
u = np.array([0, 0, 1, 1, 1])          # both users have fewer rows than chunk=6
try:
    order, g = group_sizes(u, 6)
    check('user with fewer rows than chunk does not raise', True)
    check('produces one short group per user', list(g) == [2, 3], f'got {list(g)}')
except IndexError as e:
    check('user with fewer rows than chunk does not raise', False, f'IndexError: {e}')

print("\n=== invariants across chunk sizes and shapes ===")
rng = np.random.default_rng(0)
for trial in range(6):
    users = np.repeat(np.arange(60), rng.integers(1, 40, 60))
    rng.shuffle(users)
    users = np.sort(users, kind='stable')
    for ch in (None, 1, 2, 4, 6, 7, 10, 20, 1000):
        order, g = group_sizes(users, ch)
        ok = (g.sum() == len(users) and (g > 0).all() and len(order) == len(users)
              and sorted(order.tolist()) == list(range(len(users))))
        if not ok:
            check(f'invariants hold (trial {trial}, chunk={ch})', False,
                  f'sum={g.sum()} rows={len(users)}'); break
    else:
        continue
    break
else:
    check('sizes sum to row count, no empty groups, order is a permutation', True)

print("\n=== remainder folding, not stub groups ===")
order, g = group_sizes(np.repeat([0], 13), 6)
check('13 rows at chunk 6 -> [6, 7] not [6, 6, 1]', list(g) == [6, 7], f'got {list(g)}')
check('no group of size 1 from folding', 1 not in list(g))

print("\n=== chunking preserves chronological order within a user ===")
users = np.array([1, 1, 1, 0, 0, 0])
order, g = group_sizes(users, 2)
check('stable sort keeps original row order within each user',
      list(order) == [3, 4, 5, 0, 1, 2], f'got {list(order)}')

print("\n=== real train split ===")
import loader
S = loader.load_agent()
for ch in (None, 4, 6, 7, 10, 20):
    order, g = group_sizes(S['train'].users, ch)
    if g.sum() != len(S['train']):
        check(f'real train data, chunk={ch}', False, f'{g.sum()} != {len(S["train"])}'); break
else:
    check('real train split partitions correctly at every chunk size', True)

print("\n=== invalid input rejected ===")
for bad in (0, -1, 2.5):
    try:
        group_sizes(np.array([0, 0]), bad); check(f'chunk={bad} rejected', False, 'no error')
    except ValueError:
        check(f'chunk={bad} rejected', True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
