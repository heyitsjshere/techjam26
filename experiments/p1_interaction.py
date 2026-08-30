"""Item 1. Close the delta arithmetic.

The Phase 1 marginals (+0.0033 chunking, +0.0022 rank_xendcg) summed to +0.0055
against a measured +0.0025. Two defects, both real:

  (a) both were SINGLE-SEED, against a seed std of 0.0004;
  (b) they were measured from DIFFERENT origins -- chunking from
      lambdarank/no-chunk (0.5988), rank_xendcg from binary/chunk=6 (0.6022).
      Neither origin was FM, so the two were never additive to begin with.

This is the 2x2 at 5 seeds per cell, from one common origin, so the knowledge
base records an interaction rather than two incompatible marginals.
"""
import sys, os, warnings, json; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, LOG
import loader, diagnostics, features as F
import numpy as np

S=loader.load_agent(); D=diagnostics.load_unbiased_diag()
store=F.FeatureStore().fit(S['train'])
B=('base5','duration','dur_feats')
C=dict(metric='None', eval_at=[5], learning_rate=0.05, num_leaves=63,
       min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9,
       bagging_freq=1, max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0)
SEEDS=range(5); FM=0.6016

def cell(obj, chunk):
    ps=[]
    for sd in SEEDS:
        r,_,_=run(f'ix_{obj}_chunk={chunk}_s{sd}',
                  f'2x2 interaction cell: objective={obj}, chunk={chunk}.',
                  blocks=B, params=dict(C,objective=obj), chunk=chunk,
                  store=store, splits=S, diag=D, seed=sd)
        ps.append(r['primary'])
    a=np.array(ps); return a.mean(), a.std(ddof=1)

print("2x2, 5 seeds per cell, features held at base5+duration+dur_feats\n")
res={}
for obj in ('lambdarank','rank_xendcg'):
    for ch in (None,6):
        m,s=cell(obj,ch); res[(obj,ch)]=(m,s)
        print(f"  {obj:<12s} chunk={str(ch):<4s}  {m:.4f} +/- {s:.4f}  (vs FM {m-FM:+.4f})")
bm,bs=cell('binary',None); res[('binary',None)]=(bm,bs)
print(f"  {'binary':<12s} chunk={'n/a':<4s}  {bm:.4f} +/- {bs:.4f}  (vs FM {bm-FM:+.4f})   [grouping is a no-op]")

ln,lc=res[('lambdarank',None)][0],res[('lambdarank',6)][0]
xn,xc=res[('rank_xendcg',None)][0],res[('rank_xendcg',6)][0]
print(f"""
=== the 2x2 (primary) ===
                     chunk=None   chunk=6     effect of chunking
  lambdarank           {ln:.4f}     {lc:.4f}      {lc-ln:+.4f}
  rank_xendcg          {xn:.4f}     {xc:.4f}      {xc-xn:+.4f}
  effect of xendcg     {xn-ln:+.4f}     {xc-lc:+.4f}

  interaction (non-additivity) = {(xc-xn)-(lc-ln):+.4f}
  sum of marginals from the lambdarank/None corner = {(lc-ln)+(xn-ln):+.4f}
  actual joint effect                              = {xc-ln:+.4f}
  overcount from naively adding marginals          = {((lc-ln)+(xn-ln))-(xc-ln):+.4f}

  vs official FM ({FM:.4f}): best cell {xc-FM:+.4f}
""")
with open(LOG,'a') as fh:
    fh.write(json.dumps(dict(name='PHASE1_INTERACTION_2x2',
        hypothesis='Quantify the chunking x objective interaction at 5 seeds from a '
                   'common origin; the two Phase 1 marginals were single-seed and '
                   'measured from different baselines.',
        cells={f'{o}|{c}': dict(mean=round(v[0],5), std=round(v[1],5))
               for (o,c),v in res.items()},
        interaction=round((xc-xn)-(lc-ln),5),
        marginal_sum=round((lc-ln)+(xn-ln),5),
        joint=round(xc-ln,5)))+'\n')
