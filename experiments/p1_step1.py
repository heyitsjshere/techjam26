"""Step 1. Isolated loss-function delta: LightGBM lambdarank (group=user) vs
the official pointwise FM, on the SAME 5 fields + duration_ms. Nothing else
changes, so the delta is attributable to the objective and the model class."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, show, BASELINE
import loader, diagnostics, features as F

S = loader.load_agent()
D = diagnostics.load_unbiased_diag()
store = F.FeatureStore().fit(S['train'])
print(f"train {len(S['train']):,d} | valid {len(S['valid']):,d} | diag {len(D):,d}")
print(f"official FM valid primary = {BASELINE}\n")

P = dict(objective='lambdarank', metric='None', eval_at=[5],
         learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
         feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
         max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0)

rec, booster, _ = run('step1_lambdarank_group=user',
                      'Metrics are within-user ranking metrics but FM optimises '
                      'pointwise logloss. Aligning the objective to the metric '
                      'should be the single largest available jump.',
                      blocks=('base5', 'duration'), params=P, chunk=None,
                      store=store, splits=S, diag=D)
show(rec)
print(f"\ngroups={rec['n_groups']:,d}  rows/group={rec['rows_per_group']}  "
      f"features={rec['n_features']}  best_iter={rec['best_iter']}")
print("curve:", rec['valid_curve'][:12])
