"""Tier A action space.

SEEDING BOUNDARY -- read this before editing.

This file lists what the agent CAN DO. It deliberately carries no measured
outcome for any action: no delta, no ranking, no "this one works". Every move
Phase 1 explored is present and selectable, including the ones Phase 1 found to
be dead ends, because the agent must measure them itself.

Rationale: Innovation & Problem Insight (20%) is scored on what the agent
identified as worth trying and why. A run log that opens with an inherited
answer scores nothing on that axis. The agent is given the same structural
facts a human would read off the data (see briefing.py) and must derive its own
candidate ordering from them.

Descriptions below state what an action DOES and what it would COST, never what
it is worth. "Cheap" and "expensive" are wall-clock statements, not value
statements.
"""

MODELS = {
    'fm': 'Factorization machine, pointwise logloss over embedded categorical '
          'fields. This is the reference implementation the official baseline uses.',
    'lightgbm': 'Gradient-boosted trees. Supports pointwise, pairwise and '
                'listwise objectives over the same feature frame.',
}

OBJECTIVES = {
    'binary': 'Pointwise logistic loss. Scores each impression independently; '
              'no notion of the list it sits in.',
    'lambdarank': 'Pairwise, with a gradient weighted by the NDCG gain of '
                  'swapping each pair. Uses the training group definition, and '
                  'applies a position discount over the training list length.',
    'rank_xendcg': 'Listwise softmax cross-entropy over each training group. '
                   'Uses the training group definition; no position discount.',
}

# How train rows are cut into ranking groups. Only used by ranking objectives;
# a no-op for pointwise ones.
GROUP_CHUNKS = [None, 4, 6, 7, 10, 20]
GROUP_CHUNK_DOC = (
    'None = one group per user over the whole train window. An integer caps '
    'group size at that many consecutive (chronological) rows per user, '
    'producing more, smaller groups. Cost is identical either way.')

FEATURE_BLOCKS = {
    'base5': 'user_id, video_id, author_id, tab, dur_bucket as categoricals. '
             'The five fields the official baseline uses.',
    'duration': 'Raw duration_ms.',
    'dur_feats': 'log duration, and duration percentile rank within the user list.',
    'item_agg': 'Train-window video and author rates: long-view rate, click '
                'rate, impression count. Out-of-fold encoded.',
    'user_agg': 'Train-window user rates: long-view, click, like. Out-of-fold '
                'encoded. Note these are constant across a given user rows.',
    'cross_agg': 'user x author affinity, user x tag affinity, and duration '
                 'relative to the user historical long-viewed duration.',
    'cf': 'Truncated-SVD factorisation of the train-window user x item '
          'long-view matrix; candidate scored by match to the user profile. '
          'Costs ~10s to fit once, then cached.',
}

PARAM_GRID = {
    'learning_rate': [0.02, 0.03, 0.05, 0.08],
    'num_leaves': [31, 63, 127, 255],
    'min_data_in_leaf': [20, 50, 100, 300],
    'feature_fraction': [0.6, 0.9, 1.0],
    'bagging_fraction': [0.7, 0.9, 1.0],
    'max_cat_threshold': [16, 64, 128, 512],
    'cat_smooth': [1.0, 10.0, 50.0, 200.0],
    'lambda_l2': [0.1, 1.0, 10.0, 50.0],
    'lambdarank_truncation_level': [5, 6, 10, 15, 30],
}

# Train-set shaping. Applies to which rows are used and how they are weighted.
TRAIN_SHAPING = {
    'min_date': 'Drop train rows before this date (20220408..20220421).',
    'recency_decay': 'Exponentially weight rows by age, decay per 100 date units.',
    'sample_weight': 'Uniform if unset.',
}

ENSEMBLE = {
    'rank_average': 'Rank-average the valid scores of several fitted members. '
                    'Members may differ in seed, objective, or feature blocks.',
}

# --- what the agent may NOT do (hard constraints, not judgement calls) ---
FORBIDDEN = {
    'is_rand': 'Constant 0 across both standard logs, so it carries no '
               'information by construction. Excluded from the frame.',
    'same_row_outcomes': 'long_view, play_time_ms, is_click, is_like, is_follow, '
                         'is_comment, is_forward, is_hate, profile_stay_time, '
                         'comment_stay_time, is_profile_enter are outcomes of the '
                         'impression that produced the label. Dropped at load.',
    'log_random_training': 'The randomised-exposure log is a read-only '
                           'diagnostic. Never trained on, never in selection.',
    'test_window': 'Unreachable. See src/firewall.py.',
    'loo_encoding': 'Unreachable in agent mode. Out-of-fold only. See agent/guards.py.',
}


def default_spec():
    """Iteration 0: reproduce the official baseline. Not a suggestion of where
    to go next -- it is the reference point every later delta is measured from."""
    return {
        'model': 'fm', 'objective': 'binary', 'group_chunk': None,
        'feature_blocks': ['base5'], 'params': {'k': 16, 'lr': 0.001},
        'seeds': [0],
    }


def validate(spec):
    """Structural validation only. Says nothing about whether a spec is a good
    idea -- that judgement belongs to the agent, and being wrong is informative."""
    errs = []
    if spec.get('model') not in MODELS:
        errs.append(f"model must be one of {sorted(MODELS)}")
    if spec.get('model') == 'lightgbm' and spec.get('objective') not in OBJECTIVES:
        errs.append(f"objective must be one of {sorted(OBJECTIVES)}")
    if spec.get('group_chunk') not in GROUP_CHUNKS:
        errs.append(f"group_chunk must be one of {GROUP_CHUNKS}")
    blocks = spec.get('feature_blocks') or []
    unknown = [b for b in blocks if b not in FEATURE_BLOCKS]
    if unknown:
        errs.append(f"unknown feature blocks: {unknown}")
    if not blocks:
        errs.append("feature_blocks must be non-empty")
    if 'encoding' in spec:
        errs.append("spec must not carry an encoding field; out-of-fold is enforced")
    return errs
