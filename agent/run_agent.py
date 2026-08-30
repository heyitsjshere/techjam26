"""Entry point.

    python agent/run_agent.py --mode dev    --backend stub       # plumbing only
    python agent/run_agent.py --mode dev    --backend anthropic
    python agent/run_agent.py --mode scored --backend anthropic

A scored run refuses the stub backend. Dev and scored runs write to separate
log files, because the boundary between them is what makes the
zero-intervention claim on the Autonomy axis true.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from controller import Controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dev', 'scored'], default='dev')
    ap.add_argument('--backend', choices=['stub', 'anthropic'], default='stub')
    # Must track proposer.DEFAULT_MODEL. A stale literal here silently
    # overrode the backend default and ran two scored attempts on the wrong
    # model; the exact string is a graded deliverable field.
    from proposer import DEFAULT_MODEL
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--max-iters', type=int, default=50)
    ap.add_argument('--run-id', default=None)
    a = ap.parse_args()

    if a.backend == 'stub':
        from proposer import StubBackend
        backend = StubBackend()
    else:
        from proposer import AnthropicBackend
        backend = AnthropicBackend(model=a.model)

    run_id = a.run_id or f'{a.mode}-{a.backend}'
    path = os.path.join(ROOT, 'reports', f'runlog_{a.mode}_{run_id}.jsonl')
    c = Controller(backend, path, run_id, mode=a.mode, max_iters=a.max_iters)
    summary = c.run()
    print(f"\n=== {a.mode} run complete ===")
    for k in ('iterations_used', 'manual_interventions', 'total_tokens',
              'agent_wall_clock_s', 'converged_at_iteration', 'convergence_reason'):
        print(f"  {k:<26s} {summary.get(k)}")
    print(f"  final {summary.get('final_metrics')}")
    print(f"  log: {path}")


if __name__ == '__main__':
    main()
