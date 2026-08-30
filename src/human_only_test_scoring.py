"""HUMAN ONLY. The single route to test labels. NOT IMPORTABLE BY THE AGENT.

Policy (POLICY.md sections 1, 12, 14): test is scored EXACTLY ONCE, by a human,
after the agent has converged and locked its designated submission. Its purpose
is to detect row_id misalignment. The result CANNOT change the submission.

Those two clauses used to be promises. They are now mechanisms:

  --lock   Records the designated submission's SHA-256, size, row count and the
           current git commit into reports/SUBMISSION_LOCK.json. Must be run
           BEFORE scoring. This is what makes "the result cannot change the
           submission" checkable: the submission is fingerprinted while the test
           score is still unknown.

  --score  Refuses unless a lock exists AND the file still hashes to the locked
           value, so a submission edited after locking cannot be scored as
           though it were the locked one. On success it writes a one-shot marker
           to reports/TEST_SCORED_ONCE.json and REFUSES every subsequent run.

Nothing in src/ imports this module, and it refuses to run without an explicit
confirmation flag, so an agent shelling out cannot trigger it.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'kuairand-starter-kit'))

from evaluate import evaluate                # noqa: E402
from firewall import TEST_START, TEST_END    # noqa: E402
import loader                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(ROOT, 'reports', 'SUBMISSION_LOCK.json')
ONCE_PATH = os.path.join(ROOT, 'reports', 'TEST_SCORED_ONCE.json')


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_test_split_HUMAN_ONLY(data_dir=loader.DATA_DIR):
    df = loader._read_standard_logs(data_dir)
    return loader._slice(df, TEST_START, TEST_END)


def do_lock(path):
    if os.path.exists(LOCK_PATH):
        sys.exit(f'REFUSED: a submission is already locked ({LOCK_PATH}). '
                 f'Delete it deliberately if you genuinely mean to re-lock.')
    with open(path, newline='') as fh:
        rows = sum(1 for _ in fh) - 1
    lock = {'submission': os.path.abspath(path), 'sha256': _sha256(path),
            'bytes': os.path.getsize(path), 'rows': rows,
            'locked_at': _now(), 'git_commit': _git_head()}
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    with open(LOCK_PATH, 'w') as fh:
        json.dump(lock, fh, indent=1)
    print(f'LOCKED at {lock["locked_at"]}')
    print(f'  sha256 {lock["sha256"]}')
    print(f'  {lock["rows"]:,d} rows, commit {lock["git_commit"]}')
    print('\nThe submission is now fingerprinted while the test score is still '
          'unknown. Any later edit will be refused by --score.')


def do_score(path, data_dir):
    if os.path.exists(ONCE_PATH):
        prev = json.load(open(ONCE_PATH))
        sys.exit(
            f'REFUSED: the test set has already been scored once, at '
            f'{prev["scored_at"]} (primary {prev["primary"]}). Policy allows '
            f'exactly one scoring. The marker is {ONCE_PATH}; removing it to '
            f'score again would be the intervention this mechanism exists to '
            f'prevent, and would have to be disclosed.')
    if not os.path.exists(LOCK_PATH):
        sys.exit(f'REFUSED: no submission lock at {LOCK_PATH}. Run --lock first, '
                 f'so the submission is fingerprinted BEFORE the test score is '
                 f'known.')
    lock = json.load(open(LOCK_PATH))
    actual = _sha256(path)
    if actual != lock['sha256']:
        sys.exit(f'REFUSED: this file does not match the locked submission.\n'
                 f'  locked  {lock["sha256"]}\n  actual  {actual}\n'
                 f'The submission changed after it was locked.')

    rows = load_test_split_HUMAN_ONLY(data_dir)
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        head = next(r)
        assert head == ['row_id', 'user_id', 'video_id', 'score'], head
        recs = list(r)
    assert len(recs) == len(rows), f'{len(recs)} rows vs test {len(rows)}'
    mis = sum(1 for i, rec in enumerate(recs)
              if int(rec[0]) != i
              or int(rec[1]) != rows['user_id'].iat[i]
              or int(rec[2]) != rows['video_id'].iat[i])
    print(f'alignment mismatches: {mis}   <- this is what the one shot is FOR')
    res = evaluate(rows['user_id'].tolist(),
                   (rows['long_view'].to_numpy() != 0).astype(int).tolist(),
                   [float(rec[3]) for rec in recs])
    print(f'TEST  GAUC {res["GAUC"]:.4f} | nDCG@5 {res["nDCG@5"]:.4f} | '
          f'primary {res["primary"]:.4f} | delta vs FM {res["primary"]-0.5946:+.4f}')

    marker = {'scored_at': _now(), 'submission_sha256': actual,
              'locked_at': lock['locked_at'], 'git_commit_at_lock': lock['git_commit'],
              'git_commit_at_score': _git_head(),
              'alignment_mismatches': mis,
              'GAUC': round(res['GAUC'], 5), 'nDCG@5': round(res['nDCG@5'], 5),
              'primary': round(res['primary'], 5),
              'delta_vs_fm_test': round(res['primary'] - 0.5946, 5)}
    with open(ONCE_PATH, 'w') as fh:
        json.dump(marker, fh, indent=1)
    print(f'\nOne-shot marker written to {ONCE_PATH}. Further scoring is refused.')
    print('This result CANNOT change the submission. Recorded as-is.')


def main():
    ap = argparse.ArgumentParser(description='Human-only one-shot test scoring.')
    ap.add_argument('submission')
    ap.add_argument('--i-am-a-human-and-the-submission-is-locked',
                    action='store_true', dest='confirmed', required=True)
    ap.add_argument('--data_dir', default=loader.DATA_DIR)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--lock', action='store_true',
                   help='fingerprint the submission before scoring')
    g.add_argument('--score', action='store_true',
                   help='score once against the locked fingerprint')
    a = ap.parse_args()
    if not a.confirmed:
        sys.exit('refused: confirmation flag required')
    (do_lock(a.submission) if a.lock else do_score(a.submission, a.data_dir))


if __name__ == '__main__':
    main()
