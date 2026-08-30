"""Feature cache keyed by spec hash. An unchanged block is never recomputed.

The key covers the block name, the block's implementation version, the split,
and a fingerprint of the underlying rows, so a cache hit is only possible when
the produced frame would be byte-identical.
"""
import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'cache', 'blocks')

# Bump a block's version when its implementation changes, so stale frames from
# an older definition can never be served.
BLOCK_VERSIONS = {
    'base5': 3, 'duration': 3, 'item_agg': 3, 'cross_agg': 3,
    'dur_feats': 3, 'user_agg': 3, 'cf': 3,
}


def split_fingerprint(split):
    h = hashlib.sha256()
    h.update(split.name.encode())
    h.update(np.ascontiguousarray(split.users).tobytes())
    h.update(np.ascontiguousarray(split.dates).tobytes())
    h.update(np.ascontiguousarray(split.y).tobytes())
    return h.hexdigest()[:16]


def spec_hash(block, split, extra=None):
    payload = json.dumps({'block': block,
                          'version': BLOCK_VERSIONS.get(block, 0),
                          'split': split_fingerprint(split),
                          'extra': extra or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class BlockCache:
    def __init__(self, root=CACHE_DIR):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.hits = self.misses = 0

    def get_or_build(self, block, split, build_fn, extra=None):
        key = spec_hash(block, split, extra)
        path = os.path.join(self.root, f'{block}.{key}.pkl')
        if os.path.exists(path):
            self.hits += 1
            with open(path, 'rb') as fh:
                return pickle.load(fh)
        self.misses += 1
        df = build_fn()
        with open(path, 'wb') as fh:
            pickle.dump(df, fh, protocol=5)
        return df

    def stats(self):
        return {'block_cache_hits': self.hits, 'block_cache_misses': self.misses}
