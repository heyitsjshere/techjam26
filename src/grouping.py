"""Ranking-group construction. ONE implementation, imported by every caller.

This existed twice -- once in experiments/harness.py and once in
agent/executor.py -- and the two copies diverged. The executor's copy evaluated
`blocks[-1]` before checking whether `blocks` was empty, so every user with
fewer rows than the chunk size raised IndexError. That killed exactly the specs
with `group_chunk` set, which is the structural move the agent most needs to be
able to run. Found by the first end-to-end dev run.

Duplication was the actual defect; the index error was a symptom. Hence one
implementation, here, with the edge case under test.
"""
import numpy as np


def group_sizes(users, chunk=None):
    """Contiguous LightGBM group counts for `users`.

    Returns (order, sizes) where `order` sorts rows into contiguous per-user
    blocks and `sizes` gives each group's row count, summing to len(users).

    Sorting is stable, so a user's rows keep their chronological file order and
    each chunk is a consecutive time-slice of that user's impressions.

    `chunk=None` puts one group per user. An integer caps group size, which is
    how the training list length is matched to the evaluation list length. A
    remainder is folded into the final block rather than left as a stub, since a
    1-row group contributes no pairs. A user with fewer rows than `chunk`
    becomes a single short group -- this is the case that broke the duplicated
    copy, and it is common: train users have a median of 31 rows but the
    distribution has a long left tail.
    """
    users = np.asarray(users)
    order = np.argsort(users, kind='stable')
    _, counts = np.unique(users[order], return_counts=True)
    if chunk is None:
        return order, counts.astype(np.int32)
    if not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError(f'chunk must be a positive integer or None, got {chunk!r}')

    sizes = []
    for c in counts.tolist():
        n, rem = divmod(c, chunk)
        blocks = [chunk] * n
        if rem:
            if blocks:
                blocks[-1] += rem          # fold remainder into the last block
            else:
                blocks = [rem]             # user has fewer rows than one chunk
        sizes.extend(blocks)

    g = np.array(sizes, np.int32)
    assert g.sum() == len(users), f'group sizes {g.sum()} != {len(users)} rows'
    assert (g > 0).all(), 'produced an empty group'
    return order, g
