"""Step 10. User-history / interest modelling as a CF feature.

Organizers' unexplored direction #2: the base fields use no behaviour sequence
at all. This is the CPU stand-in for DIN/SIM -- factorise the train-window
user x item long-view matrix and score a candidate by how well it matches what
that user actually long-viewed.

Why this is not another dead aggregate: it varies WITHIN a user's list (unlike
user rates) AND it is personalised (unlike video popularity, which the video_id
categorical already subsumes).
"""
import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, show
import loader, diagnostics, features as F

S = loader.load_agent(); D = diagnostics.load_unbiased_diag()
store = F.FeatureStore().fit(S['train'])
P = dict(objective='rank_xendcg', metric='None', eval_at=[5], learning_rate=0.05,
         num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
         bagging_fraction=0.9, bagging_freq=1, max_cat_threshold=64,
         cat_smooth=10.0, lambda_l2=1.0)
ref = 0.6046
for blocks, name, hyp in [
    (('base5','duration','dur_feats','cf'), 'step10_cf_k64',
     'Personalised CF affinity from the train-window long-view matrix. The one '
     'feature family not subsumed by the video_id categorical.'),
    (('base5','duration','dur_feats','cf','cross_agg'), 'step10_cf_plus_cross',
     'CF plus the coarse author/tag affinities, in case they are complementary.'),
    (('duration','dur_feats','cf'), 'step10_cf_no_ids',
     'CF without any ID categorical: does the factorisation carry the item '
     'information the video_id categorical was providing?'),
]:
    rec, _, _ = run(name, hyp, blocks=blocks, params=P, chunk=6,
                    store=store, splits=S, diag=D)
    show(rec, ref=ref)
