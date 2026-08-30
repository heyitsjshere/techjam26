"""Phase 1 experiment harness. One variable per run, isolated delta recorded.

Scoring is always the organizers' `evaluate.py` via `metrics.score`, on valid.
The unbiased random-exposure diagnostic is reported alongside but never enters
selection, early stopping, or convergence.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import lightgbm as lgb
import numpy as np
import pandas as pd

import diagnostics
import features as F
import loader
import metrics
from grouping import group_sizes

LOG = os.path.join(ROOT, 'reports', 'phase1_log.jsonl')
BASELINE = 0.6016            # official FM valid primary
NTHREAD = max(1, (os.cpu_count() or 4) - 1)


class EvalPyStopper:
    """Early stopping on evaluate.py, not LightGBM's ndcg. They disagree on
    zero-positive groups and 30.3% of valid users are all-negative."""

    def __init__(self, Xv, valid_split, every=10, patience=5):
        self.Xv, self.v, self.every, self.patience = Xv, valid_split, every, patience
        self.best, self.best_iter, self.bad, self.curve = -1.0, 0, 0, []

    def __call__(self, env):
        it = env.iteration + 1
        if it % self.every and it != env.end_iteration:
            return
        p = env.model.predict(self.Xv, num_iteration=it)
        s = metrics.score(self.v, p)['primary']
        self.curve.append((it, round(s, 5)))
        if s > self.best + 1e-5:
            self.best, self.best_iter, self.bad = s, it, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                raise lgb.callback.EarlyStopException(self.best_iter, [])


def run(name, hypothesis, blocks, params, chunk=None, num_round=1200,
        store=None, splits=None, diag=None, log=True, seed=0):
    t0 = time.time()
    splits = splits or loader.load_agent()
    store = store or F.FeatureStore().fit(splits['train'])
    tr, va = splits['train'], splits['valid']

    Xtr, Xva = store.build(tr, blocks), store.build(va, blocks)
    cats = [c for c in Xtr.columns if c in F.CATEGORICAL]
    for c in cats:
        Xtr[c] = Xtr[c].astype('category')
        Xva[c] = Xva[c].astype(pd.CategoricalDtype(Xtr[c].cat.categories))

    order, grp = group_sizes(tr.users, chunk)
    p = dict(params, num_threads=NTHREAD, seed=seed, verbosity=-1)
    ds = lgb.Dataset(Xtr.iloc[order], label=tr.y[order], group=grp,
                     categorical_feature=cats, free_raw_data=False)
    stop = EvalPyStopper(Xva, va)
    booster = lgb.train(p, ds, num_boost_round=num_round, callbacks=[stop])

    pred = booster.predict(Xva, num_iteration=stop.best_iter or booster.best_iteration)
    res = metrics.score(va, pred)

    rec = {
        'name': name, 'hypothesis': hypothesis, 'blocks': list(blocks),
        'objective': params.get('objective'), 'chunk': chunk,
        'n_features': Xtr.shape[1], 'n_groups': int(len(grp)),
        'rows_per_group': round(len(tr) / len(grp), 2),
        'best_iter': stop.best_iter, 'params': {k: v for k, v in params.items()},
        'GAUC': round(res['GAUC'], 5), 'nDCG@5': round(res['nDCG@5'], 5),
        'primary': round(res['primary'], 5),
        'delta_vs_fm': round(res['primary'] - BASELINE, 5),
        'valid_curve': stop.curve, 'seconds': round(time.time() - t0, 1),
    }

    if diag is not None:
        Xd = store.build(diag, blocks)
        for c in cats:
            Xd[c] = Xd[c].astype(pd.CategoricalDtype(Xtr[c].cat.categories))
        rec.update(diagnostics.diag_report(
            diag, booster.predict(Xd, num_iteration=stop.best_iter or booster.best_iteration)))
        rec = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}

    if log:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    return rec, booster, store


def show(rec, ref=None):
    d = f" | vs ref {rec['primary']-ref:+.4f}" if ref is not None else ""
    dg = (f" | DIAG {rec.get('DIAG_primary', float('nan')):.4f}"
          if 'DIAG_primary' in rec else "")
    print(f"{rec['name']:<40s} GAUC {rec['GAUC']:.4f} nDCG@5 {rec['nDCG@5']:.4f} "
          f"primary {rec['primary']:.4f} | vs FM {rec['delta_vs_fm']:+.4f}{d}{dg} "
          f"| {rec['best_iter']:>4d} it {rec['seconds']:>5.1f}s")
