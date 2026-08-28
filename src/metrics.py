"""The agent's only scoring entry point.

Every experiment is scored by the organizers' `evaluate.py`, never by
LightGBM's internal ndcg. The two disagree: LightGBM drops zero-positive
groups, while `evaluate.py` scores them 0.0 and keeps them in the mean. Since
30.3% of valid users are all-negative, using LightGBM's number for selection
would optimise a metric the organizers do not compute.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'kuairand-starter-kit'))

from evaluate import evaluate as _official_evaluate   # noqa: E402
from firewall import assert_agent_safe                # noqa: E402

BASELINE_VALID = {'GAUC': 0.6674, 'nDCG@5': 0.5357, 'primary': 0.6016}


def score(split, scores, where='metrics.score'):
    """Score a Split. Lock 2 fires here if test-window data ever arrives."""
    assert_agent_safe(split.name, split.min_date, split.max_date, where)
    if len(scores) != len(split):
        raise ValueError(
            f"{where}: got {len(scores)} scores for {len(split)} rows")
    r = _official_evaluate(split.users.tolist(), split.y.tolist(),
                           list(map(float, scores)))
    r['delta_primary'] = r['primary'] - BASELINE_VALID['primary']
    return r


def fmt(r, label=''):
    return (f"{label:<34s} GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | "
            f"primary {r['primary']:.4f} | delta {r['delta_primary']:+.4f}")
