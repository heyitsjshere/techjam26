"""Three end-to-end dev runs, no intervention. Not scored; failures are free."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agent'))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from controller import Controller
from proposer import AnthropicBackend

rows = []
for n in (1, 2, 3):
    rid = f'dev{n}'
    path = os.path.join(ROOT, 'reports', f'runlog_dev_{rid}.jsonl')
    if os.path.exists(path):
        os.remove(path)
    print(f"\n{'='*70}\nDEV RUN {n}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        c = Controller(AnthropicBackend(), path, rid, mode='dev', max_iters=50)
        summary = c.run()
    except Exception as e:
        print(f"RUN {n} DIED: {type(e).__name__}: {e}", flush=True)
        rows.append({'run': n, 'died': f'{type(e).__name__}: {e}'})
        continue
    recs = [json.loads(l) for l in open(path)]
    its = [r for r in recs if r['kind'] == 'ITERATION']

    def first(pred):
        for r in its:
            sp = r.get('spec') or {}
            if r.get('ran_experiment') and pred(sp):
                return r['iteration']
        return None

    chunk_it = first(lambda s: s.get('group_chunk') is not None)
    listwise_it = first(lambda s: s.get('objective') == 'rank_xendcg')
    best = summary['final_metrics']['valid_primary']
    rows.append({
        'run': n,
        'chunking_found': chunk_it is not None, 'chunking_iter': chunk_it,
        'listwise_found': listwise_it is not None, 'listwise_iter': listwise_it,
        'final_primary': best,
        'delta': summary['final_metrics']['delta_vs_fm'],
        'iters_at_convergence': summary['converged_at_iteration'],
        'iters_used': summary['iterations_used'],
        'interventions': summary['manual_interventions'],
        'tokens': summary['total_tokens'],
        'wall_s': round(time.time() - t0),
        'reason': summary['convergence_reason'],
    })
    print(json.dumps(rows[-1], indent=1), flush=True)

with open(os.path.join(ROOT, 'reports', 'dev_runs_summary.json'), 'w') as fh:
    json.dump(rows, fh, indent=1)

print(f"\n\n{'='*118}\nDEV RUN SUMMARY\n{'='*118}")
h = (f"{'run':>4} {'chunking':>9} {'@it':>4} {'listwise':>9} {'@it':>4} "
     f"{'final':>8} {'delta':>8} {'conv@':>6} {'used':>5} {'interv':>7} {'tokens':>8} {'wall':>6}")
print(h); print('-' * len(h))
for r in rows:
    if r.get('died'):
        print(f"{r['run']:>4}  DIED: {r['died'][:90]}"); continue
    print(f"{r['run']:>4} {str(r['chunking_found']):>9} {str(r['chunking_iter']):>4} "
          f"{str(r['listwise_found']):>9} {str(r['listwise_iter']):>4} "
          f"{r['final_primary']:>8.5f} {r['delta']:>+8.5f} {r['iters_at_convergence']:>6} "
          f"{r['iters_used']:>5} {r['interventions']:>7} {r['tokens']:>8,d} {r['wall_s']:>5}s")
ok = [r for r in rows if not r.get('died')]
print(f"\nchunking found in {sum(r['chunking_found'] for r in ok)}/3   "
      f"listwise found in {sum(r['listwise_found'] for r in ok)}/3   "
      f"total interventions {sum(r['interventions'] for r in ok)}")
