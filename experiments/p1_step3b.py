"""Step 3b. Grouping re-check for the winning objective. Group size was tuned
under lambdarank; rank_xendcg is a different loss and may want a different list
length. One variable: chunk, holding objective fixed at rank_xendcg."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, show
import loader, diagnostics, features as F

S = loader.load_agent(); D = diagnostics.load_unbiased_diag()
store = F.FeatureStore().fit(S['train'])
P = dict(objective='rank_xendcg', metric='None', eval_at=[5],
         learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
         feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
         max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0)
for ch in (None, 20, 10, 6, 4):
    rec, _, _ = run(f'step3b_xendcg_chunk={ch}',
                    f'Does the listwise loss want the same ~6-row lists that '
                    f'repaired lambdarank? chunk={ch}.',
                    blocks=('base5', 'duration'), params=P, chunk=ch,
                    store=store, splits=S, diag=D)
    show(rec, ref=0.6022)
