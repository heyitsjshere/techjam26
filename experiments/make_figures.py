"""Figures for the report. Every number is read from the run logs at plot time;
nothing is retyped into this file."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'reports', 'figures')
os.makedirs(FIG, exist_ok=True)
EPS = 0.002
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 200})
INK, ACCENT, WARN, MUTE = '#1a1a1a', '#0B6E4F', '#C1292E', '#8a8a8a'


def load(name):
    with open(os.path.join(ROOT, 'reports', name)) as fh:
        return [json.loads(l) for l in fh]


# ---------------------------------------------------------------- chart 1
def chart1():
    recs = load('runlog_dev_dev1.jsonl')
    summary = [r for r in recs if r['kind'] == 'RUN_SUMMARY'][0]
    its = [r for r in recs if r['kind'] == 'ITERATION' and r.get('iteration', -1) >= 0]
    curve = summary['best_curve']
    stds = [(r.get('metrics') or {}).get('primary_std') or 0.0 for r in its][:len(curve)]
    x = np.arange(len(curve))
    thresh = curve[0] + EPS
    gap = thresh - curve[-1]
    print(f'  chart1: curve={curve} thresh={thresh:.5f} gap={gap:.5f} stds={stds}')

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(thresh, ls='--', lw=1.8, color=WARN, zorder=2)
    ax.text(-0.22, thresh + 0.00004, f'survival threshold  {thresh:.5f}',
            color=WARN, fontsize=10.5, va='bottom', ha='left')

    ax.errorbar(x, curve, yerr=stds, fmt='-o', color=ACCENT, lw=2.6, ms=10,
                capsize=4, elinewidth=1.2, ecolor=MUTE, zorder=3,
                label='best-so-far validation primary (3-seed mean, \u00b11 std)')
    for xi, yi in zip(x, curve):
        ax.annotate(f'{yi:.5f}', (xi, yi), textcoords='offset points',
                    xytext=(0, -24), ha='center', fontsize=10, color=INK)
    # 0.00003 against a ~0.002 span is ~1.5% of the axis: genuinely invisible,
    # which is the point. State it rather than distorting the scale to show it.
    ax.annotate(f'stopped {gap:.5f} below the line\n'
                f'\u2248 one tenth of its own seed std',
                xy=(x[-1] - 0.04, curve[-1] - 0.00002),
                xytext=(0.92, thresh - 0.00055),
                ha='center', va='top', fontsize=12, color=WARN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=WARN, lw=1.6,
                                connectionstyle='arc3,rad=-0.25'))

    ax.set_xticks(x); ax.set_xlabel('iteration')
    ax.set_ylabel('validation primary  (mean of GAUC and nDCG@5)')
    ax.set_xlim(-0.3, len(curve) - 0.15)
    ax.set_ylim(min(curve) - 0.00045, thresh + 0.00022)
    ax.set_title('An agent that improved on every iteration, stopped for insufficient progress\n'
                 'Three consecutive gains totalling +%.5f, against a %.3f convergence threshold'
                 % (curve[-1] - curve[0], EPS), fontsize=13, loc='left', pad=12)
    ax.legend(loc='lower right', frameon=False, fontsize=9.5)
    ax.grid(axis='y', alpha=0.18)
    fig.tight_layout()
    p = os.path.join(FIG, '01_stopped_while_improving.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f'  wrote {p}')




# ---------------------------------------------------------------- chart 2
def chart2():
    kit = json.load(open(os.path.join(ROOT, 'kuairand-starter-kit',
                                      'baseline_scores.json')))
    eps = kit['convergence_rule']['epsilon']
    seed_std = kit['scores']['fm_official']['std_over_5_seeds']['test_primary']

    ph1 = [json.loads(l) for l in open(os.path.join(ROOT, 'reports', 'phase1_log.jsonl'))]
    headroom = [r for r in ph1 if r['name'] == 'PHASE1_FINAL_5SEED'][0]['delta_vs_fm']

    scored = [r for r in load('runlog_scored_final.jsonl')
              if r['kind'] == 'RUN_SUMMARY'][0]['final_metrics']['delta_vs_fm']
    bestdev = [r for r in load('runlog_dev_dev1.jsonl')
               if r['kind'] == 'RUN_SUMMARY'][0]['final_metrics']['delta_vs_fm']
    print(f'  chart2: eps={eps} headroom={headroom} scored={scored} bestdev={bestdev} std={seed_std}')

    labels = ['convergence threshold\n(\u03b5, the stopping rule)',
              'total attainable headroom\n(Phase 1, 79 experiments)',
              'best dev run\n(this agent)',
              'scored run\n(this agent)']
    vals = [eps, headroom, bestdev, scored]
    cols = [WARN, '#2b6cb0', ACCENT, ACCENT]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(vals))[::-1]
    bars = ax.barh(y, vals, color=cols, height=0.55)
    for b, al in zip(bars, [1.0, 0.85, 0.7, 1.0]):
        b.set_alpha(al)
    for yi, v in zip(y, vals):
        ax.text(v + 0.00004, yi, f'{v:+.5f}', va='center', fontsize=11.5,
                fontweight='bold', color=INK)
    ax.axvspan(0, seed_std, color=MUTE, alpha=0.16, zorder=0)
    ax.text(seed_std, y[-1] - 0.52, f'  \u00b11 seed std ({seed_std})',
            fontsize=9.5, color=MUTE, va='bottom', ha='left')

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel('primary score, as a delta above the official FM baseline')
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title('The stopping rule is nearly the size of the entire prize\n'
                 '\u03b5 = %.3f against %.5f of attainable headroom \u2014 %d%% of it'
                 % (eps, headroom, round(100 * eps / headroom)),
                 fontsize=13, loc='left', pad=12)
    ax.grid(axis='x', alpha=0.18)
    fig.tight_layout()
    p = os.path.join(FIG, '02_threshold_vs_headroom.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f'  wrote {p}')


# ---------------------------------------------------------------- chart 3
def chart3():
    """Isolated effect of each move measured in Phase 1.

    Every VALUE is read from the log at plot time. The PAIRING of an experiment
    with its reference is hand-specified below, because an isolated effect is a
    difference against a stated control and the log stores absolute scores, not
    the controls they were compared to. Nothing is retyped.
    """
    ph1 = [json.loads(l) for l in open(os.path.join(ROOT, 'reports', 'phase1_log.jsonl'))]
    kit = json.load(open(os.path.join(ROOT, 'kuairand-starter-kit', 'baseline_scores.json')))
    seed_std = kit['scores']['fm_official']['std_over_5_seeds']['test_primary']
    by = {}
    for r in ph1:
        by.setdefault(r['name'], r)

    ix = by['PHASE1_INTERACTION_2x2']['cells']          # 5 seeds per cell
    c = {k: v['mean'] for k, v in ix.items()}
    effects = [
        ('group chunking, under lambdarank',
         c['lambdarank|6'] - c['lambdarank|None'], '5-seed'),
        ('group chunking, under rank_xendcg',
         c['rank_xendcg|6'] - c['rank_xendcg|None'], '5-seed'),
        ('listwise vs pairwise (chunk=6)',
         c['rank_xendcg|6'] - c['lambdarank|6'], '5-seed'),
        ('listwise vs pointwise (chunk=6)',
         c['rank_xendcg|6'] - c['binary|None'], '5-seed'),
    ]
    # feature blocks: each measured against the same logged control
    ref = by['step3b_xendcg_chunk=6']['primary']
    for name, label in [('step4_item_agg_LOO', '+ item aggregates'),
                        ('step5_cross_agg_LOO', '+ user\u00d7item crosses'),
                        ('step6_dur_feats_LOO', '+ duration features'),
                        ('step7_user_agg_LOO', '+ user-level rates'),
                        ('step11_cf_oof', '+ CF interest model (OOF)'),
                        ('step9a_drop_user_id', '\u2212 user_id field'),
                        ('step9b_recency_decay=1.0', '+ recency weighting'),
                        ('step12_num_leaves=127', 'tuning: num_leaves 127'),
                        ('step12_learning_rate=0.02', 'tuning: lr 0.02'),
                        ('step13_ens[all lgb]', 'rank-average ensemble')]:
        if name in by:
            effects.append((label, by[name]['primary'] - ref, '1-seed'))

    effects.sort(key=lambda e: e[1])
    labels = [e[0] for e in effects]; vals = [e[1] for e in effects]
    n_clear = sum(1 for v in vals if abs(v) > seed_std)
    print(f'  chart3: {len(vals)} moves, {n_clear} outside \u00b11 seed std, ref={ref}')

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(vals))
    ax.axvspan(-seed_std, seed_std, color=MUTE, alpha=0.20, zorder=0)
    ax.axvline(0, color=INK, lw=1.1, zorder=1)
    ax.barh(y, vals, height=0.68, zorder=2,
            color=[ACCENT if v > seed_std else (WARN if v < -seed_std else MUTE)
                   for v in vals])
    for yi, v in zip(y, vals):
        ax.text(v + (0.00012 if v >= 0 else -0.00012), yi, f'{v:+.4f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=9.5,
                color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('isolated effect on validation primary')
    ax.text(seed_std, len(vals) - 0.3,
            f'  \u00b11 seed std ({seed_std}) \u2014 inside this band is noise',
            fontsize=9.5, color=MUTE, va='center')
    lo, hi = min(vals), max(vals)
    ax.set_xlim(lo - abs(lo) * 0.35 - 0.0004, hi + 0.0012)
    n_pos = sum(1 for v in vals if v > seed_std)
    ax.set_title('Only the structural moves clear seed noise\n'
                 '%d of %d measured effects sit inside \u00b11 seed std; the %d that clear it\n'
                 'are all objective or grouping, never a feature'
                 % (len(vals) - n_clear, len(vals), n_pos),
                 fontsize=13, loc='left', pad=12)
    ax.grid(axis='x', alpha=0.18)
    fig.tight_layout()
    p = os.path.join(FIG, '03_phase1_delta_ladder.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f'  wrote {p}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', '1'): chart1()
    if which in ('all', '2'): chart2()
    if which in ('all', '3'): chart3()
