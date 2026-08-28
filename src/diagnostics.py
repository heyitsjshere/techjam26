"""Unbiased-exposure diagnostic. READ-ONLY. Never enters selection.

`log_random_4_22_to_5_08_pure.csv` is randomised-exposure traffic. Phase 0
established that roughly 80% of post-4/22 impressions were diverted into this
experiment, which is why the standard log thins from ~81k rows/day to ~18k.

Rules, locked before Phase 1:
  * Clipped strictly to 20220422-20220428 to match the valid window. Everything
    from 20220429 onward is discarded at parse time -- it overlaps the test
    window and must never be observed.
  * Reported alongside valid every iteration, as a drift/overfit signal.
  * MUST NOT enter selection. MUST NOT enter the convergence calculation.
  * MUST NOT be trained on.

Read it as: does the ranking hold up on traffic the production recommender did
not choose? A gap that widens while valid improves means the model is fitting
exposure bias rather than preference.
"""
import os
import pickle

import numpy as np
import pandas as pd

from firewall import VALID_START, VALID_END, assert_agent_safe
import loader

DIAG_NAME = 'unbiased_diag'


def load_unbiased_diag(data_dir=loader.DATA_DIR, use_cache=True):
    """Random-exposure log, clipped to the valid window. Diagnostic only."""
    cache = os.path.join(loader.CACHE_DIR, 'unbiased_diag.pkl')
    if use_cache and os.path.exists(cache):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)

    df = pd.read_csv(os.path.join(data_dir, loader.RANDOM_LOG),
                     dtype=loader.LOG_DTYPES)
    df = df[(df['date'] >= VALID_START) & (df['date'] <= VALID_END)]
    df = df.reset_index(drop=True)
    assert int(df['date'].max()) <= VALID_END, 'diagnostic escaped valid window'

    split = loader._to_split(DIAG_NAME, df)
    assert_agent_safe(split.name, split.min_date, split.max_date,
                      'diagnostics.load_unbiased_diag')
    os.makedirs(loader.CACHE_DIR, exist_ok=True)
    with open(cache, 'wb') as fh:
        pickle.dump(split, fh, protocol=5)
    return split


def diag_report(split, scores):
    """Metrics under the DIAG_ prefix, so they cannot be mistaken for valid."""
    import metrics
    r = metrics.score(split, scores, where='diagnostics.diag_report')
    return {f'DIAG_{k}': v for k, v in r.items()
            if k in ('GAUC', 'nDCG@5', 'primary')}
