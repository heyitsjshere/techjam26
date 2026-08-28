"""Step 13. Rank-average ensemble. Everything above says the model is saturated
on this feature set, so the only remaining gain is variance reduction across
decorrelated fits.

The organizers' FM is deliberately NOT in this ensemble: reproducing it would
mean calling the starter kit's `data.load()`, which reads the test window into
the process. Firewall first. FM is re-implemented against the agent loader
instead, so the ensemble still gets a genuinely different model class.
"""
import sys, os, warnings, json, time; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kuairand-starter-kit'))
import harness, loader, diagnostics, features as F, metrics
import lightgbm as lgb, numpy as np, pandas as pd
from scipy.stats import rankdata

S=loader.load_agent(); D=diagnostics.load_unbiased_diag()
store=F.FeatureStore().fit(S['train'])
B=('base5','duration','dur_feats')
C=dict(metric='None', eval_at=[5], learning_rate=0.05, num_leaves=63,
       min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9,
       bagging_freq=1, max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0,
       num_threads=harness.NTHREAD, verbosity=-1)

Xtr=store.build(S['train'],B); Xva=store.build(S['valid'],B)
cats=[c for c in Xtr.columns if c in F.CATEGORICAL]
for c in cats:
    Xtr[c]=Xtr[c].astype('category')
    Xva[c]=Xva[c].astype(pd.CategoricalDtype(Xtr[c].cat.categories))

members={}
for obj,chunk,seeds in (('rank_xendcg',6,(0,1,2)),('lambdarank',6,(0,)),('binary',None,(0,))):
    for sd in seeds:
        order,grp=harness.group_sizes(S['train'].users,chunk)
        p=dict(C,objective=obj,seed=sd,bagging_seed=sd,feature_fraction_seed=sd)
        ds=lgb.Dataset(Xtr.iloc[order],label=S['train'].y[order],group=grp,
                       categorical_feature=cats,free_raw_data=False)
        st=harness.EvalPyStopper(Xva,S['valid'])
        b=lgb.train(p,ds,num_boost_round=1500,callbacks=[st])
        pr=b.predict(Xva,num_iteration=st.best_iter)
        members[f'{obj}_s{sd}']=pr
        print(f"  member {obj}_s{sd:<2d} primary {metrics.score(S['valid'],pr)['primary']:.4f}")

# --- FM re-implemented on the agent loader, for model-class diversity ---
import baseline as BM
enc={}
vocabs=[{} for _ in range(5)]
def raw(X):
    return [X['user_id'].to_numpy(),X['video_id'].to_numpy(),
            X['author_id'].to_numpy(),X['tab'].to_numpy(),X['dur_bucket'].to_numpy()]
tr_raw=raw(Xtr); va_raw=raw(Xva)
dims=[]; offs=[0]
codes=[]
for i,col in enumerate(tr_raw):
    u,inv=np.unique(col,return_inverse=True)
    vocabs[i]={v:j for j,v in enumerate(u)}
    codes.append(inv); dims.append(len(u)+1)
offs=np.cumsum([0]+dims[:-1])
Xtr_fm=np.stack([c+offs[i] for i,c in enumerate(codes)],1).astype(np.int32)
Xva_fm=np.stack([np.array([vocabs[i].get(v,len(vocabs[i])) for v in col])+offs[i]
                 for i,col in enumerate(va_raw)],1).astype(np.int32)
m=BM.FM(int(sum(dims)),k=16,lr=0.001,seed=0); rng=np.random.default_rng(0)
best,state,bad=-1,None,0
for ep in range(40):
    idx=rng.permutation(len(S['train'].y))
    for i in range(0,len(idx),8192):
        m.step(Xtr_fm[idx[i:i+8192]], S['train'].y[idx[i:i+8192]].astype(np.float32))
    pv=metrics.score(S['valid'],m.predict(Xva_fm))['primary']
    if pv>best+1e-5: best,bad,state=pv,0,(m.V.copy(),m.W.copy(),m.b)
    else:
        bad+=1
        if bad>=4: break
m.V,m.W,m.b=state
members['fm']=m.predict(Xva_fm)
print(f"  member fm      primary {best:.4f}")

def blend(names,w=None):
    R=np.stack([rankdata(members[n])/len(members[n]) for n in names])
    w=np.ones(len(names)) if w is None else np.asarray(w,float)
    return (R*w[:,None]).sum(0)/w.sum()

print("\n=== rank-average ensembles ===")
for name,ns in [('xendcg x3 seeds',['rank_xendcg_s0','rank_xendcg_s1','rank_xendcg_s2']),
                ('xendcg x3 + lambdarank',['rank_xendcg_s0','rank_xendcg_s1','rank_xendcg_s2','lambdarank_s0']),
                ('xendcg x3 + binary',['rank_xendcg_s0','rank_xendcg_s1','rank_xendcg_s2','binary_s0']),
                ('all lgb',['rank_xendcg_s0','rank_xendcg_s1','rank_xendcg_s2','lambdarank_s0','binary_s0']),
                ('all lgb + fm',list(members))]:
    r=metrics.score(S['valid'],blend(ns))
    rec=dict(name=f'step13_ens[{name}]',hypothesis='Rank-average of decorrelated fits; '
             'the only lever left once features, capacity and tuning are all flat.',
             members=ns,GAUC=round(r['GAUC'],5),**{'nDCG@5':round(r['nDCG@5'],5)},
             primary=round(r['primary'],5),delta_vs_fm=round(r['primary']-0.6016,5))
    with open(harness.LOG,'a') as fh: fh.write(json.dumps(rec)+'\n')
    print(f"{name:<26s} GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
          f"primary {r['primary']:.4f} | vs FM {r['primary']-0.6016:+.4f} | vs best single {r['primary']-0.6046:+.4f}")
