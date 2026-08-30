"""Proposer: the LLM in the loop. Emits a ranked slate of candidate experiments.

The contract is strict, because the contract is what makes the Innovation axis
measurable. Every candidate carries a hypothesis, a mechanistic rationale, an
expected gain in primary units, and a derivation for that number citing at
least one briefing fact by key.

A candidate with no cited fact still executes if it ranks, but its gain is
discounted and the discount is recorded, on the stated ground that an expected
gain with no derivation is an assertion rather than a prediction. The controller
has NO opinion about which fact is cited.

The slate is produced through the SDK's structured-output helper, so a malformed
proposal is impossible by construction rather than caught by a parser. That
matters here: an unattended scored run cannot afford to lose iterations to JSON
formatting.
"""
import json
import os
import random
import time
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DEFAULT_MODEL = 'claude-opus-5'
MAX_ATTEMPTS = 3
BASE_BACKOFF_S = 2.0

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
    'ONE_CHANGE_PER_SPEC': 'A spec is scored as a whole; bundling FEATURE BLOCK changes destroys attribution. Objective and grouping are exempt.',
    'CONVERGENCE_MECHANICS': 'Window reading: best(t) - best(t-3) <= 0.002 stops the run; gains accumulate inside the window.',
    'SEED_NOISE': 'Every spec is run at 3 seeds; differences below ~0.0008 are seed noise.',
}


# ---------------- structured output schema ----------------
class Params(BaseModel):
    learning_rate: Optional[float] = None
    num_leaves: Optional[int] = None
    min_data_in_leaf: Optional[int] = None
    feature_fraction: Optional[float] = None
    bagging_fraction: Optional[float] = None
    max_cat_threshold: Optional[int] = None
    cat_smooth: Optional[float] = None
    lambda_l2: Optional[float] = None
    lambdarank_truncation_level: Optional[int] = None


class BlockChoice(BaseModel):
    """A feature block, and why it is in THIS spec.

    The justification is required. Without it the proposer can cite a fact in
    its prose and then include blocks that contradict it -- which is exactly
    what happened before this field existed. Naming a reason per block forces
    the choice to be made rather than defaulted into, and it is also the part of
    the run log that the Innovation axis is scored on.
    """
    block: str
    justification: str = Field(
        ..., description='Why this block belongs in this spec, given the '
                         'briefing. One sentence. "It might help" is not a '
                         'justification; name the mechanism or the fact.')


class Spec(BaseModel):
    model: Literal['fm', 'lightgbm']
    objective: Optional[Literal['binary', 'lambdarank', 'rank_xendcg']] = None
    group_chunk: Optional[int] = Field(None, description='None, 4, 6, 7, 10 or 20')
    feature_blocks: List[BlockChoice]
    params: Params = Params()
    seeds: List[int] = [0]
    recency_decay: Optional[float] = None
    min_date: Optional[int] = None


class Candidate(BaseModel):
    hypothesis: str
    rationale: str
    expected_gain: float
    expected_gain_derivation: str
    tier: Literal['A', 'B']
    spec: Spec


class Slate(BaseModel):
    candidates: List[Candidate]


SYSTEM = """You are an autonomous ML research agent working on a recommender
ranking benchmark. You propose experiments; a deterministic harness executes
them and returns metrics. You never see the test set.

Each turn you return 3 to 6 candidate experiments. For each one:
  hypothesis   - what will be tried, in one sentence
  rationale    - why it might work, mechanistically
  expected_gain - your honest prediction of the change in validation primary
  expected_gain_derivation - how you arrived at that number. Cite at least one
                 FACT KEY in square brackets, e.g. [GROUP_SHAPE_MISMATCH], and
                 say how that fact leads to your number.
  tier         - "A" for a parameterised spec the harness runs directly
  spec         - the experiment configuration. Every feature block you include
                 must carry its own `justification` naming why THAT block belongs
                 in THIS spec given the briefing. A block you cannot justify is a
                 block you should not be including.

Rank honestly. Your expected_gain values determine execution order, so
systematically inflating them costs you iterations on things that do not work.
An expected gain of 0.000 is a legitimate prediction for a candidate you want to
rule out cheaply; say so in the derivation.

Think hard about ORDER. You have 50 iterations and a convergence rule that ends
the run if validation primary does not improve by more than 0.002 across 3
consecutive iterations. Three unproductive iterations in a row will end your run
whether or not you were anywhere near a real ceiling. So spend your early
iterations on whatever you believe has the largest chance of a real effect --
not on whatever is easiest to specify, and not on the cheapest thing to rule
out. Reason from the structural facts in the briefing about where a real effect
could come from, before you reason about parameters."""


class ProposerError(RuntimeError):
    pass


class Proposer:
    def __init__(self, backend):
        self.backend = backend

    def propose(self, briefing, action_space_doc, history, stall, forced_high_variance):
        user = self._prompt(briefing, action_space_doc, history, stall,
                            forced_high_variance)
        slate, usage, recovery = self.backend.complete(SYSTEM, user)
        cands = [c.model_dump() for c in slate.candidates]
        if not cands:
            raise ProposerError('proposer returned an empty slate')
        for c in cands:
            c['spec'] = {k: v for k, v in c['spec'].items() if v is not None}
            blocks = c['spec'].get('feature_blocks') or []
            c['spec']['block_justifications'] = {
                b['block']: b['justification'] for b in blocks}
            c['spec']['feature_blocks'] = [b['block'] for b in blocks]
            c['spec']['params'] = {k: v for k, v in (c['spec'].get('params') or {}).items()
                                   if v is not None}
            c['cited_facts'] = [k for k in FACT_KEYS
                                if f'[{k}]' in (c.get('expected_gain_derivation') or '')]
        return cands, usage, recovery

    @staticmethod
    def _prompt(briefing, action_space_doc, history, stall, forced):
        parts = [briefing,
                 '\n=== ACTION SPACE ===\n' + action_space_doc,
                 '\n=== FACT KEYS you may cite ===\n' +
                 '\n'.join(f'  [{k}] {v}' for k, v in FACT_KEYS.items())]
        parts.append('\n=== WHAT YOU HAVE ALREADY TRIED ===\n' +
                     (json.dumps(history, indent=1, default=str) if history
                      else '(nothing yet)'))
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
    last_recovery = []

    """Anthropic Messages API with structured output and logged backoff.

    The API key is read from os.environ ONLY. It is never written to a file,
    never logged, never printed, and never passed as an argument. If it is
    absent we fail immediately with an actionable message rather than making a
    request that would 401.

    Retries are ours, not the SDK's (`max_retries=0`), so that every attempt is
    visible as a recovery event in the run log. Retryable: 429, 5xx, connection
    errors. Not retryable: 400/401/403/404, which are bugs or config problems and
    will not fix themselves.
    """

    RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)

    def __init__(self, model=DEFAULT_MODEL, max_tokens=16000, effort='high'):
        import anthropic
        if not os.environ.get('ANTHROPIC_API_KEY'):
            raise RuntimeError(
                'ANTHROPIC_API_KEY is not set in the environment. The agent reads '
                'it from os.environ only and will not prompt for or store a key. '
                'Export it in the shell that launches the run, then retry.')
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        self._anthropic = anthropic
        # An identity-linked API key must name the workspace it acts in. The
        # workspace id is an identifier, not a credential, but it is read from
        # the environment alongside the key so neither is ever written to a file
        # by this code.
        headers = {}
        ws = os.environ.get('ANTHROPIC_WORKSPACE_ID')
        if ws:
            headers['anthropic-workspace-id'] = ws
        self.workspace_id = ws
        # SDK retries disabled so ours are the only ones and all are logged.
        self.client = anthropic.Anthropic(max_retries=0,
                                          default_headers=headers or None)

    def complete(self, system, user):
        A = self._anthropic
        recovery = []
        self.last_recovery = recovery
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                r = self.client.messages.parse(
                    model=self.model, max_tokens=self.max_tokens,
                    system=system,
                    messages=[{'role': 'user', 'content': user}],
                    thinking={'type': 'adaptive'},
                    output_config={'effort': self.effort},
                    output_format=Slate,
                )
                if getattr(r, 'stop_reason', None) == 'refusal':
                    raise ProposerError(
                        f'model refused: {getattr(r, "stop_details", None)}')
                usage = self._usage(r)
                return r.parsed_output, usage, recovery
            except (A.BadRequestError, A.AuthenticationError,
                    A.PermissionDeniedError, A.NotFoundError) as e:
                # Not retryable: a bug or a config problem, not transient.
                recovery.append({'attempt': attempt, 'error': type(e).__name__,
                                 'status': getattr(e, 'status_code', None),
                                 'retryable': False, 'action': 'aborting proposal'})
                if 'anthropic-workspace-id' in str(e) and not self.workspace_id:
                    raise ProposerError(
                        'This API key is identity-linked and must name the '
                        'workspace it acts in, but ANTHROPIC_WORKSPACE_ID is not '
                        'set. Find the id in the Console (Settings -> Workspaces; '
                        'it looks like wrkspc_...) and export it alongside the '
                        'key, then retry.') from e
                raise ProposerError(f'{type(e).__name__}: {e}') from e
            except (A.RateLimitError, A.APIStatusError, A.APIConnectionError) as e:
                status = getattr(e, 'status_code', None)
                retryable = (isinstance(e, (A.RateLimitError, A.APIConnectionError))
                             or (status in self.RETRYABLE_STATUS))
                if not retryable or attempt == MAX_ATTEMPTS:
                    recovery.append({'attempt': attempt, 'error': type(e).__name__,
                                     'status': status, 'retryable': retryable,
                                     'action': 'giving up; controller routes around'})
                    raise ProposerError(f'{type(e).__name__}: {e}') from e
                delay = BASE_BACKOFF_S * (2 ** (attempt - 1))
                if isinstance(e, A.RateLimitError):
                    try:
                        delay = max(delay, float(e.response.headers.get('retry-after', 0)))
                    except Exception:
                        pass
                delay += random.uniform(0, 0.5 * delay)      # jitter
                recovery.append({'attempt': attempt, 'error': type(e).__name__,
                                 'status': status, 'retryable': True,
                                 'action': f'backoff {delay:.1f}s, retry '
                                           f'{attempt + 1}/{MAX_ATTEMPTS}'})
                time.sleep(delay)
        raise ProposerError('exhausted retries')

    @staticmethod
    def _usage(r):
        u = r.usage
        return {'input_tokens': u.input_tokens, 'output_tokens': u.output_tokens,
                'cache_read_input_tokens': getattr(u, 'cache_read_input_tokens', 0) or 0,
                'cache_creation_input_tokens': getattr(u, 'cache_creation_input_tokens', 0) or 0,
                'model': r.model}


class StubBackend:
    last_recovery = []

    """HARNESS TEST STUB. Plumbing only; refused in a scored run.

    Deliberately encodes no conclusion from the manual probe -- it cycles
    arbitrary specs so the loop, guards, cache, logging and recovery paths can be
    exercised without an API key and without seeding an answer.
    """

    def __init__(self):
        self.n = 0

    def complete(self, system, user):
        import itertools
        pool = [('binary', None, ['base5']), ('lambdarank', 20, ['base5', 'duration']),
                ('rank_xendcg', None, ['base5', 'user_agg']),
                ('binary', 6, ['base5', 'duration', 'item_agg']),
                ('lambdarank', 4, ['base5', 'cf'])]
        picks = list(itertools.islice(itertools.cycle(pool), self.n, self.n + 3))
        self.n += 3
        slate = Slate(candidates=[
            Candidate(hypothesis=f'stub candidate {o}/{c}',
                      rationale='harness plumbing test, not a real hypothesis',
                      expected_gain=round(0.01 / (i + 1), 4),
                      expected_gain_derivation='stub, no derivation [POSITIVE_RATE]',
                      tier='A',
                      spec=Spec(model='lightgbm', objective=o, group_chunk=c,
                                feature_blocks=[BlockChoice(block=x, justification='stub') for x in b],
                                seeds=[0]))
            for i, (o, c, b) in enumerate(picks)])
        return slate, {'input_tokens': 0, 'output_tokens': 0,
                       'cache_read_input_tokens': 0,
                       'cache_creation_input_tokens': 0, 'model': 'stub'}, []
