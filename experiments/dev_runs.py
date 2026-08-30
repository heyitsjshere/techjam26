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
    stds = [r['primary_std'] for r in its if r.get('primary_std') is not None]
    rows.append({
        'conv_window': summary.get('converged_at_window'),
        'conv_per_iter': summary.get('converged_at_per_iteration'),
        'std_mean': round(sum(stds)/len(stds), 5) if stds else None,
        'std_max': round(max(stds), 5) if stds else None,
        'best_curve': summary.get('best_curve'),
        'repeats_rejected': summary.get('repeat_specs_rejected'),
        'distinct_specs': summary.get('distinct_specs_evaluated'),
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
     f"{'final':>8} {'delta':>9} {'conv_win':>9} {'conv_iter':>10} {'seedstd':>8} "
     f"{'used':>5} {'rpt':>4} {'spec':>5} {'interv':>7} {'tokens':>8} {'wall':>6}")
print(h); print('-' * len(h))
for r in rows:
    if r.get('died'):
        print(f"{r['run']:>4}  DIED: {r['died'][:90]}"); continue
    print(f"{r['run']:>4} {str(r['chunking_found']):>9} {str(r['chunking_iter']):>4} "
          f"{str(r['listwise_found']):>9} {str(r['listwise_iter']):>4} "
          f"{r['final_primary']:>8.5f} {r['delta']:>+9.5f} {str(r['conv_window']):>9} "
          f"{str(r['conv_per_iter']):>10} {str(r['std_mean']):>8} "
          f"{r['iters_used']:>5} {str(r['repeats_rejected']):>4} {str(r['distinct_specs']):>5} "
          f"{r['interventions']:>7} {r['tokens']:>8,d} {r['wall_s']:>5}s")
ok = [r for r in rows if not r.get('died')]
print(f"\nchunking found in {sum(r['chunking_found'] for r in ok)}/3   "
      f"listwise found in {sum(r['listwise_found'] for r in ok)}/3   "
      f"total interventions {sum(r['interventions'] for r in ok)}")
print(f"runs reaching iteration 5+: {sum(1 for r in ok if r['iters_used'] > 5)}/3")
print(f"spec cache fired in {sum(1 for r in ok if (r['repeats_rejected'] or 0) > 0)}/3 runs "
      f"({sum(r['repeats_rejected'] or 0 for r in ok)} repeat specs rejected total)")
for r in ok:
    print(f"  run {r['run']} best-so-far curve: {r['best_curve']}")
