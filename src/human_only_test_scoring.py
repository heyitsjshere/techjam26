"""HUMAN ONLY. The single route to test labels. NOT IMPORTABLE BY THE AGENT.

Policy (locked before Phase 1, see reports/POLICY.md):
  Test is scored exactly once, by a human, after the agent has converged and
  locked its designated submission. The purpose is to detect `row_id`
  misalignment. The result CANNOT change the submission, the model, or any
  hyperparameter. If it disagrees with valid, that disagreement is reported as
  a finding, not acted upon.

Nothing in src/ imports this module. It refuses to run unless the caller passes
--i-am-a-human-and-the-submission-is-locked, so it cannot be triggered
incidentally by an agent shelling out.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'kuairand-starter-kit'))

from evaluate import evaluate            # noqa: E402
from firewall import TEST_START, TEST_END  # noqa: E402
import loader                            # noqa: E402


def load_test_split_HUMAN_ONLY(data_dir=loader.DATA_DIR):
    df = loader._read_standard_logs(data_dir)
    return loader._slice(df, TEST_START, TEST_END)


def main():
    ap = argparse.ArgumentParser(description='Human-only one-shot test scoring.')
    ap.add_argument('submission')
    ap.add_argument('--i-am-a-human-and-the-submission-is-locked',
                    action='store_true', dest='confirmed', required=True)
    ap.add_argument('--data_dir', default=loader.DATA_DIR)
    a = ap.parse_args()
    if not a.confirmed:
        sys.exit('refused: confirmation flag required')

    rows = load_test_split_HUMAN_ONLY(a.data_dir)
    import csv
    with open(a.submission, newline='') as fh:
        r = csv.reader(fh)
        head = next(r)
        assert head == ['row_id', 'user_id', 'video_id', 'score'], head
        recs = list(r)
    assert len(recs) == len(rows), f"{len(recs)} rows vs test {len(rows)}"
    mis = sum(1 for i, rec in enumerate(recs)
              if int(rec[0]) != i
              or int(rec[1]) != rows['user_id'].iat[i]
              or int(rec[2]) != rows['video_id'].iat[i])
    print(f"alignment mismatches: {mis}  (this is what the one shot is FOR)")
    res = evaluate(rows['user_id'].tolist(),
                   (rows['long_view'].to_numpy() != 0).astype(int).tolist(),
                   [float(rec[3]) for rec in recs])
    print(f"TEST  GAUC {res['GAUC']:.4f} | nDCG@5 {res['nDCG@5']:.4f} | "
          f"primary {res['primary']:.4f} | delta vs FM {res['primary']-0.5946:+.4f}")
    print("\nThis result CANNOT change the submission. Recorded as-is.")


if __name__ == '__main__':
    main()
