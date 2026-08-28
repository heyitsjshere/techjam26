"""Test-set firewall.

Policy (locked before Phase 1, see reports/POLICY.md):
  * The agent loop has NO code path to test labels. Not behind a flag, not
    behind a config key. `loader.load_agent()` refuses to construct a test
    split at all -- test rows are dropped during parsing, so they are never
    resident in the agent's process memory.
  * Selection and convergence use `valid` exclusively.
  * Test is scored exactly once, by a human, via
    `src/human_only_test_scoring.py`, after the agent has converged and locked
    its designated submission. That score cannot change the submission.

Two independent locks, so a single mistake cannot breach the firewall:
  Lock 1 (structural): the agent loader date-filters to <= VALID_END while
          parsing. Test rows never materialise.
  Lock 2 (assertion):  every Split carries its own date range, and the agent's
          evaluate wrapper asserts that range ends at or before VALID_END.
          Fires loudly if test data ever reaches it by any route.
"""

TRAIN_START, TRAIN_END = 20220408, 20220421
VALID_START, VALID_END = 20220422, 20220428
TEST_START,  TEST_END  = 20220429, 20220508

AGENT_SPLITS = ('train', 'valid')
FORBIDDEN_SPLITS = ('test',)

# Outcomes of the same impression that produced the label. Never legal as a
# same-row feature. Dropped from every split's feature frame at load time;
# reachable on the TRAIN window only, via loader.train_outcomes(), for building
# historical aggregates.
DENY_COLUMNS = (
    'long_view', 'play_time_ms', 'is_click', 'is_like', 'is_follow',
    'is_comment', 'is_forward', 'is_hate', 'profile_stay_time',
    'comment_stay_time', 'is_profile_enter',
)

# Confirmed constant 0 across both standard logs in Phase 0 -> carries no
# information. Dropped from the action space by decision, not by accident.
DEAD_COLUMNS = ('is_rand',)


class FirewallBreach(AssertionError):
    """Raised when test-window data reaches an agent-facing code path."""


def assert_agent_safe(name, min_date, max_date, where):
    """Lock 2. Every agent-facing scoring call routes through this."""
    if name in FORBIDDEN_SPLITS:
        raise FirewallBreach(
            f"FIREWALL BREACH in {where}: split named {name!r} is forbidden to "
            f"the agent. Test is scored once by a human via "
            f"src/human_only_test_scoring.py, never from the agent loop."
        )
    if max_date is not None and max_date > VALID_END:
        raise FirewallBreach(
            f"FIREWALL BREACH in {where}: split {name!r} spans "
            f"{min_date}-{max_date}, which reaches past the valid window "
            f"(ends {VALID_END}). Test-window rows must never reach the "
            f"agent's evaluate wrapper."
        )
    return True


def assert_no_deny_columns(columns, where):
    """Same-row leakage guard. Feature frames must contain no denied column."""
    bad = sorted(set(columns) & set(DENY_COLUMNS))
    if bad:
        raise FirewallBreach(
            f"LEAKAGE in {where}: feature frame contains same-row outcome "
            f"column(s) {bad}. These are outcomes of the impression that "
            f"produced the label. Historical aggregates over the train window "
            f"are legal; the raw same-row value is not."
        )
    return True
