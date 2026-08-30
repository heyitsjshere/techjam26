"""The seeding boundary is a claim about what the agent is NOT told. A claim
that is not tested erodes on the next edit, so it is tested here.

Fails if a Phase 1 measured outcome, verdict, or final configuration leaks into
anything the proposer reads.
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'agent'))
import actionspace, briefing, proposer

PASS, FAIL = [], []
def check(n, cond, msg=''):
    (PASS if cond else FAIL).append(n)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{'' if cond else ': ' + msg}")

# Everything the proposer actually sees, concatenated.
SEEN = (briefing.full_briefing() + '\n'
        + str(actionspace.MODELS) + str(actionspace.OBJECTIVES)
        + str(actionspace.GROUP_CHUNKS) + actionspace.GROUP_CHUNK_DOC
        + str(actionspace.FEATURE_BLOCKS) + str(actionspace.PARAM_GRID)
        + str(actionspace.TRAIN_SHAPING) + str(actionspace.ENSEMBLE)
        + str(actionspace.FORBIDDEN) + proposer.SYSTEM + str(proposer.FACT_KEYS))
LOW = SEEN.lower()

print("=== no Phase 1 measured delta leaks ===")
# Any Phase 1 result value that would give away an outcome.
FORBIDDEN_NUMBERS = ['0.6041', '0.6046', '0.6044', '0.6021', '0.5988', '0.6022',
                     '0.0025', '0.0033', '0.0022', '0.0030', '0.0137', '0.6039']
for n in FORBIDDEN_NUMBERS:
    check(f'{n} absent', n not in SEEN, 'leaked a Phase 1 measurement')

print("\n=== permitted numbers ARE present (briefing must still be useful) ===")
for n, why in [('0.6016', 'baseline score, public'), ('43.5', 'train group size'),
               ('5.6', 'valid group size'), ('0.8484', 'oracle ceiling, public'),
               ('0.002', 'convergence eps, public'), ('0.5950', 'organizer ablation, public')]:
    check(f'{n} present ({why})', n in SEEN, 'briefing lost a fact it needs')

print("\n=== no verdict about which move pays ===")
BAD_PHRASES = [
    'chunking works', 'chunk the groups', 'chunking fixes', 'use rank_xendcg',
    'rank_xendcg is best', 'rank_xendcg wins', 'listwise is best',
    'best config', 'final config', 'the answer is', 'we found that',
    'phase 1 found', 'phase 1 measured', 'is the winning', 'you should use',
    'the two that pay', 'proved to be', 'turned out to',
]
for p in BAD_PHRASES:
    check(f'no phrase {p!r}', p not in LOW, 'a verdict leaked into the briefing')

print("\n=== dead moves remain SELECTABLE, not removed ===")
for b in ('item_agg', 'user_agg', 'cross_agg', 'cf'):
    check(f'{b} still offered', b in actionspace.FEATURE_BLOCKS,
          'a Phase 1 dead end was removed from the action space')
check('all group chunks offered', set(actionspace.GROUP_CHUNKS) >= {None, 4, 6, 7, 10, 20})
check('all three objectives offered', set(actionspace.OBJECTIVES) == {'binary', 'lambdarank', 'rank_xendcg'})
check('recency_decay still offered', 'recency_decay' in actionspace.TRAIN_SHAPING)
check('ensembling still offered', 'rank_average' in actionspace.ENSEMBLE)

print("\n=== feature block descriptions carry no value judgement ===")
VALUE_WORDS = ['useless', 'redundant', 'no gain', 'ineffective', 'best', 'worst',
               'recommended', 'avoid', 'do not use', 'strongest', 'weakest']
for b, d in actionspace.FEATURE_BLOCKS.items():
    bad = [w for w in VALUE_WORDS if w in d.lower()]
    check(f'{b} description neutral', not bad, f'value words {bad}')

print("\n=== the structural facts the agent must reason FROM are present ===")
for frag, why in [('43.5', 'train rows per user'), ('5.6', 'valid rows per user'),
                  ('POINTWISE', 'baseline objective stated'),
                  ('within-user', 'metric form stated'),
                  ('constant', 'user-constant consequence stated'),
                  ('randomised-exposure', 'CF divergence observation present')]:
    check(f'{why}', frag.lower() in LOW)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
