"""Step 2. Isolated group-size delta. Same lambdarank config as step 1; the
ONLY change is how train rows are cut into ranking groups.

Motivation is now mechanistic, not just distributional: step 1 showed the
ranking objective is actively harmful at group=user. lambdarank optimises NDCG
over its training lists, and those lists are 43.5 rows while evaluation lists
are 5.6. If the mismatch is the cause, shrinking train groups should recover
the loss. Chunking is a no-op for pointwise objectives, so this isolates cleanly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, show, group_sizes, BASELINE
import loader, diagnostics, features as F
import numpy as np

S = loader.load_agent(); D = diagnostics.load_unbiased_diag()
store = F.FeatureStore().fit(S['train'])
P = dict(objective='lambdarank', metric='None', eval_at=[5],
         learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
         feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
         max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0)

print("=== train grouping, before and after ===")
print(f"{'chunk':>7s} {'groups':>9s} {'rows/group':>11s} {'median':>7s} {'p90':>5s}")
for ch in (None, 20, 10, 7, 6, 4):
    _, g = group_sizes(S['train'].users, ch)
    print(f"{str(ch):>7s} {len(g):>9,d} {len(S['train'])/len(g):>11.2f} "
          f"{np.median(g):>7.0f} {np.percentile(g,90):>5.0f}")
_, gv = group_sizes(S['valid'].users, None)
print(f"{'VALID':>7s} {len(gv):>9,d} {len(S['valid'])/len(gv):>11.2f} "
      f"{np.median(gv):>7.0f} {np.percentile(gv,90):>5.0f}   <- the target regime\n")

print("=== isolated delta of group size (lambdarank, base5+duration) ===")
ref = 0.5988
for ch in (20, 10, 7, 6, 4):
    rec, _, _ = run(f'step2_lambdarank_chunk={ch}',
                    f'Match train ranking-group size ({ch}) to the ~6-row lists '
                    f'evaluation actually scores.',
                    blocks=('base5', 'duration'), params=P, chunk=ch,
                    store=store, splits=S, diag=D)
    show(rec, ref=ref)
