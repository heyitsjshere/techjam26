"""Phase 1 final. Multi-seed estimate of the best configuration.

Single-seed bests are selection-biased: three xendcg seeds spanned 0.6034-0.6046,
so reporting 0.6046 would be reporting seed luck. The organizers quote 5-seed
means with std 0.0008; this matches that convention.
"""
import sys, os, warnings, json; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, LOG
import loader, diagnostics, features as F
import numpy as np

S=loader.load_agent(); D=diagnostics.load_unbiased_diag()
store=F.FeatureStore().fit(S['train'])
B=('base5','duration','dur_feats')
P=dict(objective='rank_xendcg', metric='None', eval_at=[5], learning_rate=0.05,
       num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9,
       bagging_freq=1, max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0)
rows=[]
for sd in range(5):
    rec,_,_=run(f'final_xendcg_chunk6_seed{sd}',
                'Locked Phase 1 config, seed variance measurement.',
                blocks=B, params=P, chunk=6, store=store, splits=S, diag=D, seed=sd)
    rows.append(rec); print(f"  seed {sd}: primary {rec['primary']:.4f} "
                            f"GAUC {rec['GAUC']:.4f} nDCG@5 {rec['nDCG@5']:.4f} "
                            f"DIAG {rec['DIAG_primary']:.4f}")
g=np.array([r['GAUC'] for r in rows]); n=np.array([r['nDCG@5'] for r in rows])
p=np.array([r['primary'] for r in rows])
print(f"\n{'':22s}{'mean':>8s} {'std':>8s}")
for nm,a in (('GAUC',g),('nDCG@5',n),('primary',p)):
    print(f"  {nm:<20s}{a.mean():>8.4f} {a.std(ddof=1):>8.4f}")
print(f"\n  official FM valid primary   0.6016")
print(f"  Phase 1 delta over FM       {p.mean()-0.6016:+.4f} +/- {p.std(ddof=1):.4f}")
print(f"  best single seed            {p.max():.4f} ({p.max()-0.6016:+.4f})  <- selection-biased")
summary=dict(name='PHASE1_FINAL_5SEED', hypothesis='Honest multi-seed estimate.',
             blocks=list(B), chunk=6, params=P,
             GAUC_mean=round(g.mean(),5), GAUC_std=round(g.std(ddof=1),5),
             ndcg_mean=round(n.mean(),5), primary_mean=round(p.mean(),5),
             primary_std=round(p.std(ddof=1),5),
             delta_vs_fm=round(p.mean()-0.6016,5),
             best_single_seed=round(p.max(),5))
with open(LOG,'a') as fh: fh.write(json.dumps(summary)+'\n')
