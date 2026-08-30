"""Hard guards. These are enforcement, not convention.

Guard 1 -- OUT-OF-FOLD ENCODING ONLY.
  Phase 1 measured leave-one-out target encoding at -0.0137 against naive: LOO
  removes self-inclusion but substitutes an offset that is a deterministic
  function of the row's own label, which a tree learns to invert. Naive
  encoding leaks outright. Only out-of-fold has neither defect.

  While the agent is running, `agent_mode()` is active and any encoding other
  than 'oof' raises GuardViolation from inside the encoder itself. The Tier A
  spec schema has no encoding field, so there is no way for the agent to ask
  for one; this guard catches the case where agent-written Tier B code tries to
  call the encoder directly. LOO is unreachable, not discouraged.

Guard 2 -- MANDATORY DRIFT CHECK.
  Every feature block is checked for train-vs-valid distribution drift BEFORE
  any metric computed on it is believed or recorded. Phase 1 found both of its
  methodology bugs this way and neither was visible in the metric first: the
  leaking cross-feature scored 0.4866 (near random) with no indication of why.

  A block that fails is rejected, and the rejection is logged with its drift
  magnitude and the offending columns. No metric for a drift-failing block is
  ever written to the run log as a result.
"""
import contextlib
import threading

import numpy as np

_state = threading.local()

# Standardised mean difference above which a feature is REJECTED. Calibrated
# empirically in tests/test_guards.py against Phase 1's known-good and
# known-leaking features:
#     duration                        SMD 0.049
#     item_agg (out-of-fold)          SMD 0.107
#     cross_agg (corrected)           SMD 0.209
#     dur_rank_in_list                SMD 0.268   <- covariate shift, not leakage
#     cross_agg (naive, KNOWN LEAK)   SMD 0.769
# 0.40 sits 1.5x above the highest legitimate block and 1.9x below the known
# leak. The WARN band exists because the calibration surfaced a real distinction:
# dur_rank_in_list is a duration percentile computed WITHIN a user's list, and a
# percentile over 43.5 rows is not the same quantity as one over 5.6. That is
# genuine covariate shift caused by the group-shape mismatch, not a leak, and
# rejecting it would be a false positive. It is flagged, not blocked.
DRIFT_SMD_THRESHOLD = 0.40
DRIFT_SMD_WARN = 0.25
DRIFT_EXEMPT = frozenset({'user_id', 'video_id', 'author_id', 'tab', 'dur_bucket'})


class GuardViolation(RuntimeError):
    """A hard guard was violated. Never caught and retried -- the run halts."""


def in_agent_mode():
    return getattr(_state, 'agent', False)


@contextlib.contextmanager
def agent_mode():
    """Everything the agent executes runs inside this. Guards are live within."""
    prev = getattr(_state, 'agent', False)
    _state.agent = True
    try:
        yield
    finally:
        _state.agent = prev


def check_encoding(mode):
    """Called from inside the encoder on every encode. Guard 1."""
    if in_agent_mode() and mode != 'oof':
        raise GuardViolation(
            f"encoding={mode!r} is unreachable in agent mode. Phase 1 measured "
            f"leave-one-out at -0.0137 vs naive (label-dependent offset the tree "
            f"inverts) and naive leaks the row's own outcome. Out-of-fold only."
        )
    return mode


def drift_report(train_df, valid_df, exempt=DRIFT_EXEMPT):
    """Standardised mean difference per column, train vs valid.

    SMD is scale-free, so it compares a rate against a log-duration without a
    per-feature threshold. ID categoricals are exempt: their 'mean' is an
    arbitrary label encoding and carries no distributional meaning.
    """
    rows = []
    for c in train_df.columns:
        if c in exempt:
            rows.append(dict(feature=c, smd=0.0, exempt=True,
                             train_mean=None, valid_mean=None))
            continue
        a = np.asarray(train_df[c], np.float64)
        b = np.asarray(valid_df[c], np.float64)
        sa, sb = a.std(), b.std()
        pooled = np.sqrt((sa ** 2 + sb ** 2) / 2.0)
        smd = 0.0 if pooled < 1e-12 else abs(a.mean() - b.mean()) / pooled
        rows.append(dict(feature=c, smd=round(float(smd), 4), exempt=False,
                         warn=bool(smd > DRIFT_SMD_WARN),
                         train_mean=round(float(a.mean()), 6),
                         valid_mean=round(float(b.mean()), 6)))
    return rows


def check_drift(train_df, valid_df, threshold=DRIFT_SMD_THRESHOLD, where=''):
    """Guard 2. Returns (passed, report). The caller MUST branch on `passed`
    before computing or recording any metric."""
    rep = drift_report(train_df, valid_df)
    bad = [r for r in rep if not r['exempt'] and r['smd'] > threshold]
    warn = [r for r in rep if not r['exempt'] and r.get('warn') and r['smd'] <= threshold]
    return (not bad), dict(
        passed=not bad, threshold=threshold, warn_threshold=DRIFT_SMD_WARN,
        where=where,
        max_smd=max((r['smd'] for r in rep if not r['exempt']), default=0.0),
        failed=[r['feature'] for r in bad],
        warned=[r['feature'] for r in warn],
        failed_detail=bad, per_feature=rep,
    )
