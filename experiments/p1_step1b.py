"""Step 1 control arm. step1 changed TWO things vs FM at once: the model class
(FM -> GBDT) and the objective (pointwise -> lambdarank). Without a pointwise
GBDT on identical features, step1's delta is not attributable to either.

  delta(model)     = LGB binary   - FM
  delta(objective) = LGB lambdarank - LGB binary   <- the number we actually want
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, show, BASELINE
import loader, diagnostics, features as F

S = loader.load_agent(); D = diagnostics.load_unbiased_diag()
store = F.FeatureStore().fit(S['train'])

COMMON = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
              max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0)

rec, _, _ = run('step1b_binary_control',
                'Pointwise GBDT on identical features. Separates the model-class '
                'delta from the objective delta, which step1 confounds.',
                blocks=('base5', 'duration'),
                params=dict(COMMON, objective='binary', metric='None'),
                chunk=None, store=store, splits=S, diag=D)
show(rec)
print(f"\nFM (pointwise, embeddings) valid primary = {BASELINE:.4f}")
print(f"LGB binary     (pointwise, trees)  = {rec['primary']:.4f}  "
      f"-> delta(model class)  = {rec['primary']-BASELINE:+.4f}")
print(f"LGB lambdarank (ranking,   trees)  = 0.5988  "
      f"-> delta(objective)    = {0.5988-rec['primary']:+.4f}")
