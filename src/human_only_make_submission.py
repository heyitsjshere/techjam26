"""HUMAN ONLY. Build the submission CSV for the designated config.

NOT IMPORTABLE BY THE AGENT. This is the only place a model is asked to predict
on test-window rows, and it happens after the agent has converged and designated
its config -- never during a run.

What is and is not crossed here: the model is trained on the TRAIN window only,
exactly as the agent trained it, and feature artifacts are fitted on train only.
Test-window rows supply FEATURES for prediction. Test LABELS are never read by
this module -- they are read once, later, by human_only_test_scoring.py.

Row order is the organizers': both standard logs in fixed order, date-filtered,
positional index = row_id. Verified bit-identical to data.load() in
tests/test_firewall.py.
"""
import argparse
import csv
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'agent'))

import features as F
import loader
from executor import DEFAULT_PARAMS, NTHREAD
from firewall import TEST_START, TEST_END
from grouping import group_sizes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_test_split_HUMAN_ONLY(data_dir=loader.DATA_DIR):
    df = loader._read_standard_logs(data_dir)
    df = loader._slice(df, TEST_START, TEST_END)
    return loader._to_split('test', df), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--spec', required=True, help='JSON file with the designated spec')
    ap.add_argument('--i-am-a-human-and-the-agent-has-converged',
                    action='store_true', dest='confirmed', required=True)
    ap.add_argument('--data_dir', default=loader.DATA_DIR)
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    seeds = spec.get('seeds') or [0, 1, 2]
    blocks = tuple(spec['feature_blocks'])
    print(f"designated: {spec['model']} {spec.get('objective')} "
          f"chunk={spec.get('group_chunk')} blocks={sorted(blocks)} seeds={seeds}")

    splits = loader.load_agent()
    store = F.FeatureStore().fit(splits['train'])
    test_split, test_df = build_test_split_HUMAN_ONLY(a.data_dir)
    print(f"train {len(splits['train']):,d} rows | test {len(test_split):,d} rows")

    Xtr = store.build(splits['train'], blocks, encoding='oof')
    Xte = store.build(test_split, blocks)
    cats = [c for c in Xtr.columns if c in F.CATEGORICAL]
    for c in cats:
        Xtr[c] = Xtr[c].astype('category')
        Xte[c] = Xte[c].astype(pd.CategoricalDtype(Xtr[c].cat.categories))

    order, grp = group_sizes(splits['train'].users, spec.get('group_chunk'))
    preds = []
    for sd in seeds:
        p = dict(DEFAULT_PARAMS, **(spec.get('params') or {}))
        p.update(objective=spec['objective'], num_threads=NTHREAD, verbosity=-1,
                 seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
        ds = lgb.Dataset(Xtr.iloc[order], label=splits['train'].y[order], group=grp,
                         categorical_feature=cats, free_raw_data=False)
        n = spec.get('best_iter') or 200
        b = lgb.train(p, ds, num_boost_round=n)
        preds.append(b.predict(Xte))
        print(f"  seed {sd}: trained {n} rounds")

    score = np.mean(preds, axis=0)
    assert np.isfinite(score).all(), 'non-finite score'
    assert len(score) == len(test_split)

    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['row_id', 'user_id', 'video_id', 'score'])
        uid = test_df['user_id'].to_numpy(); vid = test_df['video_id'].to_numpy()
        for i, s in enumerate(score):
            w.writerow([i, uid[i], vid[i], f'{float(s):.6g}'])
    print(f"\nwrote {a.out}: {len(score):,d} rows, mean of {len(seeds)} seeds")
    print('Test LABELS were not read by this module.')


if __name__ == '__main__':
    main()
