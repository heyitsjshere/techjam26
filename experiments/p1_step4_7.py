"""Steps 4-7. Feature blocks, added one at a time on top of the locked model
config (rank_xendcg, chunk=6). Each row is the isolated delta of ONE block
given everything already in.

Why this is not covered by the organizers' negative result: their ablation
added static categorical ID fields (music_id, video_type, upload_type, user
demographic buckets). It never tested behavioural aggregates over the training
window. Those are a different kind of information and are untested.
"""
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

STEPS = [
 ('step4_item_agg_LOO', 'item_agg',
  'Video-side historical rates vary WITHIN a user list, so unlike user-side '
  'features they can move the within-user order. Untested by the organizers. Now with leave-one-out on train rows.'),
 ('step5_cross_agg_LOO', 'cross_agg',
  'user x item crosses (author affinity, tag affinity, duration-vs-preference) '
  'are the only way user information can affect a within-user ranking.'),
 ('step6_dur_feats_LOO', 'dur_feats',
  'long_view is watch time relative to duration, so duration bias is '
  'structural. Cheap proxy for debiasing: log duration and duration rank '
  'within the user list.'),
 ('step7_user_agg_LOO', 'user_agg',
  'DECISIVE TEST. User-level rates are CONSTANT within a user, so they cannot '
  'shift the order directly and can only act as tree gates. If the isolated '
  'delta is under 0.001, delete them from the action space.'),
]

blocks = ['base5', 'duration']
prev = 0.6044
print(f"{'locked: rank_xendcg, chunk=6':<40s} running best = {prev:.4f}\n")
for name, blk, hyp in STEPS:
    rec, _, _ = run(name, hyp, blocks=tuple(blocks + [blk]), params=P, chunk=6,
                    store=store, splits=S, diag=D)
    show(rec, ref=prev)
    print(f"    -> isolated delta of +{blk}: {rec['primary']-prev:+.4f}")
    if rec['primary'] > prev:
        blocks.append(blk); prev = rec['primary']
    else:
        print(f"    -> {blk} NOT carried forward")
print(f"\nfinal block set: {blocks}   valid primary {prev:.4f}  vs FM {prev-0.6016:+.4f}")
