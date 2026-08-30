"""Tier A executor. Turns a validated spec into metrics, or into a logged
rejection. The agent does not write this code; it emits specs and this runs them.

Order of operations is load-bearing:
    build features -> DRIFT CHECK -> (only if passed) train -> score
A block that fails drift never reaches a model, so no metric for it is ever
produced, let alone recorded. Phase 1's two methodology bugs were both invisible
in the metric and both visible in drift.
"""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'agent'))

import lightgbm as lgb
import numpy as np
import pandas as pd

import diagnostics
import features as F
import guards
import loader
import metrics
from cache import BlockCache
from grouping import group_sizes

NTHREAD = max(1, (os.cpu_count() or 4) - 1)
DEFAULT_PARAMS = dict(metric='None', eval_at=[5], learning_rate=0.05,
                      num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
                      bagging_fraction=0.9, bagging_freq=1, max_cat_threshold=64,
                      cat_smooth=10.0, lambda_l2=1.0)


class ExecutionError(RuntimeError):
    pass


class _Stopper:
    """Early stopping on evaluate.py, never on LightGBM's internal ndcg. The two
    disagree on zero-positive groups and 30.3% of valid users are all-negative."""

    def __init__(self, Xv, split, every=10, patience=5):
        self.Xv, self.v, self.every, self.patience = Xv, split, every, patience
        self.best, self.best_iter, self.bad, self.curve = -1.0, 0, 0, []

    def __call__(self, env):
        it = env.iteration + 1
        if it % self.every and it != env.end_iteration:
            return
        s = metrics.score(self.v, env.model.predict(self.Xv, num_iteration=it))['primary']
        self.curve.append((it, round(s, 5)))
        if s > self.best + 1e-5:
            self.best, self.best_iter, self.bad = s, it, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                raise lgb.callback.EarlyStopException(self.best_iter, [])


class Executor:
    def __init__(self, use_diagnostic=True):
        self.splits = loader.load_agent()
        self.store = F.FeatureStore().fit(self.splits['train'])
        self.cache = BlockCache()
        self.diag = diagnostics.load_unbiased_diag() if use_diagnostic else None

    # ---------- features, cached per block ----------
    def _frame(self, split, blocks):
        parts = []
        for b in sorted(blocks):
            parts.append(self.cache.get_or_build(
                b, split, lambda b=b: self.store.build(split, (b,), encoding='oof')))
        return pd.concat(parts, axis=1)

    # ---------- the guarded path ----------
    def run(self, spec, seeds=None):
        """Returns a dict with drift always populated, and metrics populated only
        if drift passed and training succeeded."""
        t0 = time.time()
        errs = __import__('actionspace').validate(spec)
        if errs:
            raise ExecutionError(f"invalid spec: {errs}")

        blocks = tuple(spec['feature_blocks'])
        with guards.agent_mode():
            Xtr = self._frame(self.splits['train'], blocks)
            Xva = self._frame(self.splits['valid'], blocks)

            # --- GUARD 2, before anything is trained or believed ---
            passed, drift = guards.check_drift(Xtr, Xva, where=f"blocks={list(blocks)}")
            if not passed:
                return dict(ok=False, rejected_by='drift_check', drift=drift,
                            metrics=None, diagnostic=None,
                            seconds=round(time.time() - t0, 1),
                            cache=self.cache.stats())

            if spec['model'] == 'fm':
                res = self._run_fm(Xtr, Xva, spec, seeds or spec.get('seeds', [0]))
            else:
                res = self._run_lgb(Xtr, Xva, spec, seeds or spec.get('seeds', [0]))

        res.update(ok=True, rejected_by=None, drift=drift,
                   seconds=round(time.time() - t0, 1), cache=self.cache.stats())
        return res

    # ---------- models ----------
    def _prep(self, Xtr, Xva, extra=()):
        Xtr, Xva = Xtr.copy(), Xva.copy()
        cats = [c for c in Xtr.columns if c in F.CATEGORICAL]
        for c in cats:
            Xtr[c] = Xtr[c].astype('category')
            dt = pd.CategoricalDtype(Xtr[c].cat.categories)
            Xva[c] = Xva[c].astype(dt)
            for Z in extra:
                Z[c] = Z[c].astype(dt)
        return Xtr, Xva, cats

    def _run_lgb(self, Xtr, Xva, spec, seeds):
        tr, va = self.splits['train'], self.splits['valid']
        Xd = self._frame(self.diag, tuple(spec['feature_blocks'])) if self.diag is not None else None
        Xtr, Xva, cats = self._prep(Xtr, Xva, extra=[Xd] if Xd is not None else [])
        chunk = spec.get('group_chunk')
        order, grp = group_sizes(tr.users, chunk)
        w = None
        if spec.get('recency_decay'):
            age = (tr.dates.max() - tr.dates[order]).astype(np.float64)
            w = np.exp(-spec['recency_decay'] * age / 100.0)
        preds, dpreds, its = [], [], []
        for sd in seeds:
            p = dict(DEFAULT_PARAMS, **spec.get('params', {}))
            p.update(objective=spec['objective'], num_threads=NTHREAD, verbosity=-1,
                     seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
            ds = lgb.Dataset(Xtr.iloc[order], label=tr.y[order], group=grp, weight=w,
                             categorical_feature=cats, free_raw_data=False)
            st = _Stopper(Xva, va)
            b = lgb.train(p, ds, num_boost_round=spec.get('num_round', 1500),
                          callbacks=[st])
            preds.append(b.predict(Xva, num_iteration=st.best_iter))
            its.append(st.best_iter)
            if Xd is not None:
                dpreds.append(b.predict(Xd, num_iteration=st.best_iter))
        return self._score(preds, dpreds, seeds, its, grp)

    def _run_fm(self, Xtr, Xva, spec, seeds):
        """FM against the AGENT loader. The starter kit's own data.load() reads the
        test window, so it is never called from inside the agent."""
        sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
        import baseline as BM
        tr, va = self.splits['train'], self.splits['valid']
        cols = [c for c in F.BASE5 if c in Xtr.columns]
        vocabs, codes, dims = [], [], []
        for c in cols:
            u, inv = np.unique(Xtr[c].to_numpy(), return_inverse=True)
            vocabs.append({v: j for j, v in enumerate(u)}); codes.append(inv)
            dims.append(len(u) + 1)
        offs = np.cumsum([0] + dims[:-1])
        A = np.stack([c + offs[i] for i, c in enumerate(codes)], 1).astype(np.int32)
        B = np.stack([np.array([vocabs[i].get(v, len(vocabs[i]))
                                for v in Xva[c].to_numpy()]) + offs[i]
                      for i, c in enumerate(cols)], 1).astype(np.int32)
        preds, its = [], []
        for sd in seeds:
            m = BM.FM(int(sum(dims)), k=spec.get('params', {}).get('k', 16),
                      lr=spec.get('params', {}).get('lr', 0.001), seed=sd)
            rng = np.random.default_rng(sd)
            best, state, bad = -1, None, 0
            for ep in range(spec.get('params', {}).get('epochs', 40)):
                idx = rng.permutation(len(tr.y))
                for i in range(0, len(idx), 8192):
                    m.step(A[idx[i:i + 8192]], tr.y[idx[i:i + 8192]].astype(np.float32))
                pv = metrics.score(va, m.predict(B))['primary']
                if pv > best + 1e-5:
                    best, bad, state = pv, 0, (m.V.copy(), m.W.copy(), m.b)
                else:
                    bad += 1
                    if bad >= 4:
                        break
            m.V, m.W, m.b = state
            preds.append(m.predict(B)); its.append(ep + 1)
        return self._score(preds, [], seeds, its, None)

    def _score(self, preds, dpreds, seeds, its, grp):
        va = self.splits['valid']
        per = [metrics.score(va, p) for p in preds]
        pr = np.array([r['primary'] for r in per])
        out = dict(
            metrics=dict(
                GAUC=round(float(np.mean([r['GAUC'] for r in per])), 5),
                **{'nDCG@5': round(float(np.mean([r['nDCG@5'] for r in per])), 5)},
                primary=round(float(pr.mean()), 5),
                primary_std=round(float(pr.std(ddof=1)) if len(pr) > 1 else 0.0, 5),
                n_seeds=len(seeds), per_seed=[round(x, 5) for x in pr],
                delta_vs_fm=round(float(pr.mean()) - 0.6016, 5),
                best_iters=its),
            diagnostic=None)
        if dpreds:
            dg = [diagnostics.diag_report(self.diag, p) for p in dpreds]
            out['diagnostic'] = {k: round(float(np.mean([d[k] for d in dg])), 5)
                                 for k in dg[0]}
        if grp is not None:
            out['metrics']['n_train_groups'] = int(len(grp))
            out['metrics']['rows_per_group'] = round(len(self.splits['train']) / len(grp), 2)
        return out
