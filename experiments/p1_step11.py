"""Step 11. CF profile construction, isolated. Step 10 used leave-one-out, whose
penalty step 8 measured at -0.0137 -- almost exactly step 10's -0.0138. So the CF
verdict has to be re-taken under naive and out-of-fold profiles before it counts.
"""
import sys, os, warnings, time, json; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import harness, loader, diagnostics, features as F, metrics
import lightgbm as lgb, numpy as np, pandas as pd

S=loader.load_agent(); D=diagnostics.load_unbiased_diag()
store=F.FeatureStore().fit(S['train'])
P=dict(objective='rank_xendcg', metric='None', eval_at=[5], learning_rate=0.05,
       num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9,
       bagging_freq=1, max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0,
       num_threads=harness.NTHREAD, seed=0, verbosity=-1)
BL=('base5','duration','dur_feats','cf')

def go(name, enc, hyp, blocks=BL):
    t0=time.time()
    Xtr=store.build(S['train'],blocks,encoding=enc); Xva=store.build(S['valid'],blocks)
    Xd =store.build(D,blocks)
    cats=[c for c in Xtr.columns if c in F.CATEGORICAL]
    for c in cats:
        Xtr[c]=Xtr[c].astype('category')
        for Z in (Xva,Xd): Z[c]=Z[c].astype(pd.CategoricalDtype(Xtr[c].cat.categories))
    order,grp=harness.group_sizes(S['train'].users,6)
    ds=lgb.Dataset(Xtr.iloc[order],label=S['train'].y[order],group=grp,
                   categorical_feature=cats,free_raw_data=False)
    st=harness.EvalPyStopper(Xva,S['valid'])
    b=lgb.train(P,ds,num_boost_round=1500,callbacks=[st])
    r=metrics.score(S['valid'],b.predict(Xva,num_iteration=st.best_iter))
    dg=diagnostics.diag_report(D,b.predict(Xd,num_iteration=st.best_iter))
    rec=dict(name=name,hypothesis=hyp,blocks=list(blocks),encoding=enc,
             GAUC=round(r['GAUC'],5),**{'nDCG@5':round(r['nDCG@5'],5)},
             primary=round(r['primary'],5),delta_vs_fm=round(r['primary']-0.6016,5),
             best_iter=st.best_iter,seconds=round(time.time()-t0,1),
             n_features=Xtr.shape[1],n_groups=int(len(grp)),
             rows_per_group=round(len(S['train'])/len(grp),2),
             params={k:v for k,v in P.items() if k!='num_threads'},valid_curve=st.curve,
             **{k:round(v,5) for k,v in dg.items()})
    with open(harness.LOG,'a') as fh: fh.write(json.dumps(rec)+'\n')
    harness.show(rec,ref=0.6046); return rec

print("=== CF profile construction (base5+duration+dur_feats+cf, xendcg, chunk=6) ===")
print("    reference without cf: valid 0.6046 / DIAG 0.3664\n")
for enc in ('loo','naive','oof'):
    go(f'step11_cf_{enc}', enc, f'CF user profile built with {enc} handling of the '
       f'row own contribution on train rows.')
