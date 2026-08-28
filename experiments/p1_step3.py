"""Step 3. Objective sweep at the best grouping so far (chunk=6), plus
lambdarank_truncation_level tuned against the ~6-item list length instead of
left at its default of 30.

Step 1 + step 2 together say the ranking objective only breaks even. This step
tests whether that is a truncation-level artifact before the objective question
is closed out.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, show, BASELINE
import loader, diagnostics, features as F

S = loader.load_agent(); D = diagnostics.load_unbiased_diag()
store = F.FeatureStore().fit(S['train'])
C = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
         feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
         max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0, metric='None')
B = ('base5', 'duration')
ref = 0.6022   # LGB binary control

print("=== lambdarank_truncation_level at chunk=6 (default is 30) ===")
for tl in (5, 6, 10, 15, 30):
    rec, _, _ = run(f'step3_lambdarank_trunc={tl}',
                    f'Eval lists average 5.6 items; truncation_level=30 spends '
                    f'gradient on positions that never exist. Try {tl}.',
                    blocks=B, params=dict(C, objective='lambdarank', eval_at=[5],
                                          lambdarank_truncation_level=tl),
                    chunk=6, store=store, splits=S, diag=D)
    show(rec, ref=ref)

print("\n=== objective family at chunk=6 ===")
for obj, note in (('rank_xendcg', 'Listwise cross-entropy; less sensitive to list length than lambdarank.'),
                  ('binary', 'Pointwise control. Grouping is a no-op for this objective.')):
    p = dict(C, objective=obj)
    if obj == 'rank_xendcg':
        p['eval_at'] = [5]
    rec, _, _ = run(f'step3_{obj}_chunk=6', note, blocks=B, params=p,
                    chunk=6 if obj != 'binary' else None,
                    store=store, splits=S, diag=D)
    show(rec, ref=ref)
