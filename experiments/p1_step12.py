"""Step 12. Hyperparameter sweep on the locked feature set. Structured grid, one
axis at a time from a fixed centre, so each move stays attributable."""
import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run, show
import loader, diagnostics, features as F

S=loader.load_agent(); D=diagnostics.load_unbiased_diag()
store=F.FeatureStore().fit(S['train'])
B=('base5','duration','dur_feats')
C=dict(objective='rank_xendcg', metric='None', eval_at=[5], learning_rate=0.05,
       num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9,
       bagging_freq=1, max_cat_threshold=64, cat_smooth=10.0, lambda_l2=1.0)
ref=0.6046; best=(ref,{})
AXES=[('learning_rate',[0.02,0.03,0.08]),
      ('num_leaves',[31,127,255]),
      ('min_data_in_leaf',[20,100,300]),
      ('max_cat_threshold',[16,128,512]),
      ('cat_smooth',[1.0,50.0,200.0]),
      ('lambda_l2',[0.1,10.0,50.0]),
      ('feature_fraction',[0.6,1.0])]
for ax,vals in AXES:
    print(f"\n--- {ax} (centre {C[ax]} = {ref:.4f}) ---")
    for v in vals:
        rec,_,_=run(f'step12_{ax}={v}', f'One-axis sweep: {ax}={v}.',
                    blocks=B, params=dict(C,**{ax:v}), chunk=6,
                    store=store, splits=S, diag=D)
        show(rec, ref=ref)
        if rec['primary']>best[0]: best=(rec['primary'],{ax:v})
print(f"\nbest single-axis move: {best[1]} -> {best[0]:.4f} (centre {ref:.4f})")
