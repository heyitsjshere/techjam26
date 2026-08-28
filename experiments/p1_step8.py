"""Step 8. Target-encoding scheme, isolated. Same features, same model, same
grouping -- only how the train-window aggregate is applied to train rows changes.

This is the correction to step 4. A naive train-window aggregate leaks the row's
own outcome; leave-one-out removes that but substitutes a label-dependent
offset; out-of-fold does neither. The agent must not be seeded with a feature
verdict that was really an encoding artifact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import harness
from harness import show
import loader, diagnostics, features as F
import lightgbm as lgb, numpy as np, pandas as pd, metrics, json, time

S = loader.load_agent(); D = diagnostics.load_unbiased_diag()
store = F.FeatureStore().fit(S['train'])
P = dict(objective='rank_xendcg', metric='None', eval_at=[5], learning_rate=0.05,
         num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
         bagging_fraction=0.9, bagging_freq=1, max_cat_threshold=64,
         cat_smooth=10.0, lambda_l2=1.0, num_threads=harness.NTHREAD,
         seed=0, verbosity=-1)

def go(name, blocks, enc, hyp):
    t0=time.time()
    Xtr = store.build(S['train'], blocks, encoding=enc)
    Xva = store.build(S['valid'], blocks)
    cats=[c for c in Xtr.columns if c in F.CATEGORICAL]
    for c in cats:
        Xtr[c]=Xtr[c].astype('category')
        Xva[c]=Xva[c].astype(pd.CategoricalDtype(Xtr[c].cat.categories))
    order,grp = harness.group_sizes(S['train'].users, 6)
    ds = lgb.Dataset(Xtr.iloc[order], label=S['train'].y[order], group=grp,
                     categorical_feature=cats, free_raw_data=False)
    st = harness.EvalPyStopper(Xva, S['valid'])
    b = lgb.train(P, ds, num_boost_round=1200, callbacks=[st])
    r = metrics.score(S['valid'], b.predict(Xva, num_iteration=st.best_iter))
    rec = dict(name=name, hypothesis=hyp, blocks=list(blocks), encoding=enc,
               GAUC=round(r['GAUC'],5), **{'nDCG@5':round(r['nDCG@5'],5)},
               primary=round(r['primary'],5), delta_vs_fm=round(r['primary']-0.6016,5),
               best_iter=st.best_iter, seconds=round(time.time()-t0,1),
               n_features=Xtr.shape[1], n_groups=int(len(grp)),
               rows_per_group=round(len(S['train'])/len(grp),2),
               params={k:v for k,v in P.items() if k!='num_threads'}, valid_curve=st.curve)
    with open(harness.LOG,'a') as fh: fh.write(json.dumps(rec)+'\n')
    show(rec, ref=0.6044); return rec

print("=== encoding scheme, item_agg on top of base5+duration (chunk=6, xendcg) ===")
for enc in ('naive','loo','oof'):
    go(f'step8_item_agg_enc={enc}', ('base5','duration','item_agg'), enc,
       f'Target-encoding scheme for train rows: {enc}.')

print("\n=== redundancy test: are ID categoricals already doing this work? ===")
go('step8_aggs_only_no_ids', ('duration','item_agg'), 'oof',
   'video_id is a 7.5k-level categorical the tree can fit directly, so a '
   'train-window video rate may be strictly redundant with it. Drop the IDs '
   'and keep only the aggregates: if this holds up, they encode the same thing.')
go('step8_base5_no_user_id_proxy', ('base5','duration','user_agg'), 'oof',
   'User-level rates are within-user constants. With OOF encoding, do they now '
   'earn their place as tree gates?')
