"""Step 9. The levers nobody has tested: field ablation, distribution shift, and
capacity. Features are confirmed dead; these are what is left.
"""
import sys, os, warnings, time, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import harness, loader, diagnostics, features as F, metrics
import lightgbm as lgb, numpy as np, pandas as pd

S = loader.load_agent(); D = diagnostics.load_unbiased_diag()
store = F.FeatureStore().fit(S['train'])
BASE = dict(objective='rank_xendcg', metric='None', eval_at=[5], learning_rate=0.05,
            num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
            bagging_fraction=0.9, bagging_freq=1, max_cat_threshold=64,
            cat_smooth=10.0, lambda_l2=1.0, num_threads=harness.NTHREAD,
            seed=0, verbosity=-1)

def go(name, hyp, drop=(), params=None, chunk=6, min_date=None, decay=None,
       blocks=('base5','duration','dur_feats'), ret=False):
    t0=time.time(); p = dict(BASE, **(params or {}))
    Xtr = store.build(S['train'], blocks); Xva = store.build(S['valid'], blocks)
    Xtr = Xtr.drop(columns=list(drop)); Xva = Xva.drop(columns=list(drop))
    cats=[c for c in Xtr.columns if c in F.CATEGORICAL]
    for c in cats:
        Xtr[c]=Xtr[c].astype('category')
        Xva[c]=Xva[c].astype(pd.CategoricalDtype(Xtr[c].cat.categories))
    y, users, dates = S['train'].y, S['train'].users, S['train'].dates
    keep = np.ones(len(y), bool) if min_date is None else (dates >= min_date)
    Xt, yt, ut, dt = Xtr[keep], y[keep], users[keep], dates[keep]
    order, grp = harness.group_sizes(ut, chunk)
    w = None
    if decay is not None:
        age = (dt.max() - dt[order]).astype(np.float64)
        w = np.exp(-decay * age / 100.0)
    ds = lgb.Dataset(Xt.iloc[order], label=yt[order], group=grp, weight=w,
                     categorical_feature=cats, free_raw_data=False)
    st = harness.EvalPyStopper(Xva, S['valid'])
    b = lgb.train(p, ds, num_boost_round=1500, callbacks=[st])
    pred = b.predict(Xva, num_iteration=st.best_iter)
    r = metrics.score(S['valid'], pred)
    rec = dict(name=name, hypothesis=hyp, blocks=list(blocks), dropped=list(drop),
               chunk=chunk, min_date=min_date, decay=decay,
               train_rows=int(keep.sum()), n_groups=int(len(grp)),
               GAUC=round(r['GAUC'],5), **{'nDCG@5':round(r['nDCG@5'],5)},
               primary=round(r['primary'],5), delta_vs_fm=round(r['primary']-0.6016,5),
               best_iter=st.best_iter, seconds=round(time.time()-t0,1),
               n_features=Xt.shape[1], rows_per_group=round(len(yt)/len(grp),2),
               params={k:v for k,v in p.items() if k!='num_threads'}, valid_curve=st.curve)
    with open(harness.LOG,'a') as fh: fh.write(json.dumps(rec)+'\n')
    harness.show(rec, ref=0.6046)
    return (rec, b, pred) if ret else rec

print("=== 9a. field ablation: is the 26k-level user_id categorical earning its place? ===")
go('step9a_full_reference','Reference: base5+duration+dur_feats.')
go('step9a_drop_user_id','user_id is a within-user CONSTANT: it cannot reorder a list '
   'directly, only gate. At 26,210 levels with max_cat_threshold=64 it is a prime '
   'overfitting surface. Drop it.', drop=('user_id',))
go('step9a_drop_author_id','author_id is largely collinear with video_id. Drop it.',
   drop=('author_id',))
go('step9a_drop_both_ids','Drop user_id and author_id together.',
   drop=('user_id','author_id'))

print("\n=== 9b. distribution shift: valid sits immediately after train ===")
for md in (20220412, 20220415, 20220418):
    go(f'step9b_train_from_{md}', f'Exposure regime changed at 4/22 and valid is the '
       f'week after train. Early train days may be off-distribution. Train from {md}.',
       min_date=md)
for dc in (0.5, 1.0, 2.0):
    go(f'step9b_recency_decay={dc}', f'Keep all rows but weight recent days more '
       f'(exp decay {dc} per 100 date-units), instead of hard-truncating.', decay=dc)
