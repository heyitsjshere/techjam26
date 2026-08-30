"""Proposer: the LLM in the loop. Emits a ranked slate of candidate experiments.

The contract is deliberately strict, because the contract is what makes the
Innovation axis measurable. Every candidate must carry:

  hypothesis                what will be tried, in one sentence
  rationale                 why it might work, mechanistically
  expected_gain             a number, in primary units
  expected_gain_derivation  how that number was arrived at, citing at least one
                            briefing fact by key
  spec                      a Tier A spec, or a Tier B patch

A candidate with no cited briefing fact is not rejected -- it is executed if it
ranks -- but its uncited status is recorded, and the controller discounts it, on
the stated ground that an expected gain with no derivation is an assertion
rather than a prediction. This is a mechanism for making reasoning visible. It
is NOT a ranking over which facts matter; the controller has no opinion about
which fact a candidate cites.
"""
import json

FACT_KEYS = {
    'WITHIN_USER_RANKING': 'Only within-user order affects the score.',
    'USER_CONSTANT_NO_EFFECT': 'A quantity constant across a user rows cannot reorder that user list.',
    'GROUP_SHAPE_MISMATCH': 'Train averages 43.5 rows/user; valid averages 5.6.',
    'SHORT_EVAL_LISTS': 'nDCG@5 over ~6 items is close to ordering the whole list.',
    'POINTWISE_BASELINE': 'The baseline optimises pointwise logloss; the metrics are ranking metrics.',
    'EXPOSURE_REGIME_CHANGE': '~80% of post-4/22 traffic was diverted to a randomised experiment.',
    'ORGANIZER_FEATURES_DEAD': 'Organizers measured added static fields as no gain.',
    'ORGANIZER_CAPACITY_DEAD': 'Organizers measured added embedding capacity as no gain.',
    'ORGANIZER_UNEXPLORED': 'Organizers list loss function, behaviour sequences, multi-task, duration bias as unexplored.',
    'CF_DIAGNOSTIC_DIVERGENCE': 'A CF feature scored worse on valid and better on the unbiased log.',
    'HIGH_COVERAGE': '99.9% of valid videos appear in train.',
    'CEILING': 'Perfect ranking scores 0.8484 on valid; only 57.8% of users are influenceable.',
    'POSITIVE_RATE': 'Positive rate ~0.31 on train and valid.',
}

SYSTEM = """You are an autonomous ML research agent working on a recommender
ranking benchmark. You propose experiments; a deterministic harness executes
them and returns metrics. You never see the test set.

Each turn, return STRICT JSON: {"candidates": [ ... ]}, 3 to 6 candidates,
each with keys: hypothesis, rationale, expected_gain (float, in primary units,
your honest prediction), expected_gain_derivation (string; cite at least one
FACT KEY in square brackets, e.g. [GROUP_SHAPE_MISMATCH], and say how that fact
leads to your number), tier ("A" or "B"), spec (object).

Rank honestly. Your expected_gain values determine execution order, so
systematically inflating them costs you iterations on things that do not work.
An expected gain of 0.000 is a legitimate prediction for a candidate you want
to rule out cheaply; say so.

Think about ORDER. You have 50 iterations and a convergence rule that stops the
run if validation primary does not improve by more than 0.002 across 3
consecutive iterations. Three unproductive iterations in a row will end your run
whether or not you were near a real ceiling. So spend your early iterations on
whatever you believe has the largest chance of a real effect, not on whatever is
easiest to specify."""


class ProposerError(RuntimeError):
    pass


class Proposer:
    """Backend-agnostic. `complete(system, user) -> (text, tokens_in, tokens_out)`."""

    def __init__(self, backend):
        self.backend = backend

    def propose(self, briefing, action_space_doc, history, stall, forced_high_variance):
        user = self._prompt(briefing, action_space_doc, history, stall,
                            forced_high_variance)
        text, ti, to = self.backend.complete(SYSTEM, user)
        try:
            obj = json.loads(self._strip(text))
            cands = obj['candidates']
            assert isinstance(cands, list) and cands
        except Exception as e:
            raise ProposerError(f"unparseable proposal: {e}\n{text[:500]}")
        for c in cands:
            c['cited_facts'] = [k for k in FACT_KEYS
                                if f'[{k}]' in c.get('expected_gain_derivation', '')]
        return cands, ti, to

    @staticmethod
    def _strip(t):
        t = t.strip()
        if t.startswith('```'):
            t = t.split('\n', 1)[1].rsplit('```', 1)[0]
        return t[t.index('{'):t.rindex('}') + 1]

    @staticmethod
    def _prompt(briefing, action_space_doc, history, stall, forced):
        parts = [briefing,
                 '\n=== ACTION SPACE ===\n' + action_space_doc,
                 '\n=== FACT KEYS you may cite ===\n' +
                 '\n'.join(f'  [{k}] {v}' for k, v in FACT_KEYS.items())]
        if history:
            parts.append('\n=== WHAT YOU HAVE ALREADY TRIED ===\n' +
                         json.dumps(history, indent=1, default=str))
        else:
            parts.append('\n=== WHAT YOU HAVE ALREADY TRIED ===\n(nothing yet)')
        parts.append(f'\nConsecutive non-improving iterations so far: {stall}.')
        if forced:
            parts.append(
                'The controller is FORCING A HIGH-VARIANCE MOVE this iteration. '
                'Two consecutive iterations have not improved the metric, and one '
                'more ends the run. Do not propose another marginal variation of '
                'what you have already tried. Propose the most different thing you '
                'can justify from the briefing.')
        return '\n'.join(parts)


class AnthropicBackend:
    def __init__(self, model='claude-opus-4-6', max_tokens=4000):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system, user):
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{'role': 'user', 'content': user}])
        return (r.content[0].text, r.usage.input_tokens, r.usage.output_tokens)


class StubBackend:
    """HARNESS TEST STUB. Exercises plumbing on --dry-run only.

    It deliberately does NOT encode any conclusion from the manual probe: it
    cycles arbitrary specs so the loop, guards, cache, logging and recovery
    paths can be tested without an API key and without seeding an answer.
    Never used in a scored run; the controller refuses it when mode='scored'.
    """

    def __init__(self):
        self.n = 0

    def complete(self, system, user):
        import itertools
        pool = [
            ('binary', None, ['base5']), ('lambdarank', 20, ['base5', 'duration']),
            ('rank_xendcg', None, ['base5', 'user_agg']),
            ('binary', 6, ['base5', 'duration', 'item_agg']),
            ('lambdarank', 4, ['base5', 'cf']),
        ]
        picks = list(itertools.islice(itertools.cycle(pool), self.n, self.n + 3))
        self.n += 3
        cands = [{'hypothesis': f'stub candidate {o}/{c}',
                  'rationale': 'harness plumbing test, not a real hypothesis',
                  'expected_gain': round(0.01 / (i + 1), 4),
                  'expected_gain_derivation': 'stub, no derivation [POSITIVE_RATE]',
                  'tier': 'A',
                  'spec': {'model': 'lightgbm', 'objective': o, 'group_chunk': c,
                           'feature_blocks': b, 'params': {}, 'seeds': [0]}}
                 for i, (o, c, b) in enumerate(picks)]
        return json.dumps({'candidates': cands}), 0, 0
