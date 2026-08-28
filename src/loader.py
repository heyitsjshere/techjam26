"""KuaiRand-Pure loading for the agent.

Row order is bit-identical to the organizers' `data.load()`: the two standard
logs are read in fixed order (4_08_to_4_21 first, then 4_22_to_5_08), then
date-filtered while preserving original file order. `row_id` in a submission is
the positional index into the resulting split, so this ordering is load-bearing
and is asserted against the reference loader in tests/test_firewall.py.
"""
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from firewall import (TRAIN_START, TRAIN_END, VALID_START, VALID_END,
                      DENY_COLUMNS, DEAD_COLUMNS,
                      assert_agent_safe, assert_no_deny_columns)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'kuairand-starter-kit', 'KuaiRand-Pure', 'data')
CACHE_DIR = os.path.join(ROOT, 'cache')

STANDARD_LOGS = ('log_standard_4_08_to_4_21_pure.csv',
                 'log_standard_4_22_to_5_08_pure.csv')
RANDOM_LOG = 'log_random_4_22_to_5_08_pure.csv'

LOG_DTYPES = {
    'user_id': np.int32, 'video_id': np.int32, 'date': np.int32,
    'hourmin': np.int32, 'time_ms': np.int64, 'duration_ms': np.float64,
    'tab': np.int16, 'is_rand': np.int8,
    **{c: np.int32 for c in DENY_COLUMNS},
}


@dataclass
class Split:
    """A split the agent may hold. Carries its own provenance for Lock 2."""
    name: str
    X: pd.DataFrame          # features only; never contains a DENY column
    y: np.ndarray            # long_view, 0/1
    users: np.ndarray        # user_id per row, for within-user grouping
    dates: np.ndarray        # date per row, for the firewall assertion
    meta: dict = field(default_factory=dict)

    @property
    def min_date(self): return int(self.dates.min())

    @property
    def max_date(self): return int(self.dates.max())

    def __len__(self): return len(self.y)

    def __repr__(self):
        return (f"Split({self.name!r}, rows={len(self):,d}, "
                f"users={len(np.unique(self.users)):,d}, "
                f"dates={self.min_date}-{self.max_date}, "
                f"cols={list(self.X.columns)})")


def _read_standard_logs(data_dir):
    """Both standard logs, concatenated in the organizers' fixed order."""
    return pd.concat(
        [pd.read_csv(os.path.join(data_dir, f), dtype=LOG_DTYPES)
         for f in STANDARD_LOGS],
        ignore_index=True, copy=False)


def _slice(df, lo, hi):
    """Date window, preserving original row order. Positional index = row_id."""
    return df[(df['date'] >= lo) & (df['date'] <= hi)].reset_index(drop=True)


def _to_split(name, df):
    """Strip label and every same-row outcome column out of the feature frame."""
    y = df['long_view'].to_numpy(np.int8)
    y = (y != 0).astype(np.int8)
    users = df['user_id'].to_numpy(np.int32)
    dates = df['date'].to_numpy(np.int32)
    X = df.drop(columns=[c for c in (*DENY_COLUMNS, *DEAD_COLUMNS)
                         if c in df.columns])
    assert_no_deny_columns(X.columns, f'_to_split({name!r})')
    return Split(name=name, X=X, y=y, users=users, dates=dates)


def load_agent(data_dir=DATA_DIR, use_cache=True):
    """THE agent-facing loader. Returns train and valid. Never test.

    Lock 1: rows dated past VALID_END are discarded during parsing, so test
    rows are never resident in the agent's process. There is deliberately no
    argument, flag, or config key that changes this.
    """
    cache = os.path.join(CACHE_DIR, 'agent_splits.pkl')
    if use_cache and os.path.exists(cache):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)

    df = _read_standard_logs(data_dir)
    df = df[df['date'] <= VALID_END]                      # <-- Lock 1
    splits = {'train': _to_split('train', _slice(df, TRAIN_START, TRAIN_END)),
              'valid': _to_split('valid', _slice(df, VALID_START, VALID_END))}
    for s in splits.values():
        assert_agent_safe(s.name, s.min_date, s.max_date, 'loader.load_agent')

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, 'wb') as fh:
        pickle.dump(splits, fh, protocol=5)
    return splits


def train_outcomes(data_dir=DATA_DIR, use_cache=True):
    """Denied outcome columns, TRAIN WINDOW ONLY, for historical aggregates.

    This is the one legal route to `is_click`/`long_view`/etc. It is clamped to
    the train window, so an aggregate built from it can never see a valid or
    test outcome. Join the result onto features causally.
    """
    cache = os.path.join(CACHE_DIR, 'train_outcomes.pkl')
    if use_cache and os.path.exists(cache):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)

    df = _read_standard_logs(data_dir)
    df = _slice(df, TRAIN_START, TRAIN_END)
    assert int(df['date'].max()) <= TRAIN_END, 'train_outcomes escaped train window'
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, 'wb') as fh:
        pickle.dump(df, fh, protocol=5)
    return df


def video_features(data_dir=DATA_DIR):
    basic = pd.read_csv(os.path.join(data_dir, 'video_features_basic_pure.csv'))
    stat = pd.read_csv(os.path.join(data_dir, 'video_features_statistic_pure.csv'))
    return basic, stat


def user_features(data_dir=DATA_DIR):
    return pd.read_csv(os.path.join(data_dir, 'user_features_pure.csv'))
