"""Feature blocks. Every block is a named unit so Phase 1's measured deltas map
directly onto the agent's Tier A action space.

All artifacts are fitted on the TRAIN window only, via `loader.train_outcomes()`
(clamped to <= 20220421), and joined causally onto every split.

LEAVE-ONE-OUT
-------------
A train-window aggregate applied back to train rows contains that row's own
outcome. The model then learns to trust a signal that does not exist at
evaluation time, and collapses. Measured directly in Phase 1: naive
`ua_affinity` had train mean 0.0149 vs valid mean 0.0003, a ~50x collapse, and
the model fell to 0.4866 primary -- near random.

So every count-based aggregate subtracts the row's own contribution when it is
applied to train rows (`loo=True`), and does not when applied to valid or the
diagnostic. `train_outcomes()` and `load_agent()['train']` are sliced from the
same frame in the same order, so the alignment is positional and asserted.
"""
import numpy as np
import pandas as pd

import loader
from firewall import assert_no_deny_columns

BASE5 = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
CATEGORICAL = set(BASE5)
BLOCKS = ('base5', 'duration', 'item_agg', 'cross_agg', 'dur_feats', 'user_agg', 'cf')


def _rate(pos, cnt, prior, gmean, own_pos=None, own_cnt=None):
    """Shrunk rate, optionally leaving the current row out."""
    if own_pos is not None:
        pos = pos - own_pos
        cnt = cnt - own_cnt
    return (pos + prior * gmean) / np.maximum(cnt, 0) + 0 * gmean if False else \
           (pos + prior * gmean) / (np.maximum(cnt, 0) + prior)


class _Encoder:
    """Target encoding with three selectable schemes, so the choice is measured
    rather than assumed.

      naive : train-window totals applied to every row, train included.
              LEAKS -- a train row sees its own outcome.
      loo   : subtract the row's own contribution on train rows.
              Removes self-inclusion but INVERTS: the offset -1/(cnt+prior) is a
              deterministic function of the row's own label, so a deep enough
              tree learns "slightly lower rate => positive", which is backwards
              at eval time. Measured in Phase 1, it is worse than naive.
      oof   : K-fold out-of-fold. A train row's encoding is built from the folds
              it is NOT in, so no self-inclusion and no label-dependent offset.
              This is the correct one.
    """

    def __init__(self, keys, target, n_folds=5, seed=0):
        self.tot_pos = pd.Series(target).groupby(keys).sum()
        self.tot_cnt = pd.Series(np.ones(len(keys))).groupby(keys).sum()
        rng = np.random.default_rng(seed)
        self.folds = rng.integers(0, n_folds, len(keys))
        self.n_folds = n_folds
        self.fold_pos = [pd.Series(target * (self.folds == k)).groupby(keys).sum()
                         for k in range(n_folds)]
        self.fold_cnt = [pd.Series((self.folds == k).astype(float)).groupby(keys).sum()
                         for k in range(n_folds)]

    def encode(self, keys, prior, gmean, mode, own=None):
        # Guard 1: in agent mode only out-of-fold is reachable. Enforced here,
        # inside the encoder, so agent-written Tier B code cannot route around it.
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'agent'))
        import guards
        guards.check_encoding(mode)
        pos = self.tot_pos.reindex(keys).fillna(0.0).to_numpy(np.float64)
        cnt = self.tot_cnt.reindex(keys).fillna(0.0).to_numpy(np.float64)
        if mode == 'naive' or own is None:
            pass
        elif mode == 'loo':
            pos = pos - own
            cnt = cnt - 1.0
        elif mode == 'oof':
            fp = np.zeros(len(keys)); fc = np.zeros(len(keys))
            for k in range(self.n_folds):
                m = self.folds == k
                if m.any():
                    fp[m] = self.fold_pos[k].reindex(keys[m]).fillna(0.0).to_numpy()
                    fc[m] = self.fold_cnt[k].reindex(keys[m]).fillna(0.0).to_numpy()
            pos, cnt = pos - fp, cnt - fc
        else:
            raise ValueError(mode)
        return (pos + prior * gmean) / (np.maximum(cnt, 0.0) + prior), np.maximum(cnt, 0.0)


class FeatureStore:
    def __init__(self, data_dir=loader.DATA_DIR, prior=20.0):
        self.prior, self.data_dir = prior, data_dir
        basic, _ = loader.video_features(data_dir)
        self.vid2author = dict(zip(basic['video_id'], basic['author_id']))
        tags = basic['tag'].fillna(-1).astype(str)
        self.vid2tags = {v: tuple(int(float(x)) for x in s.split(',') if x not in ('', 'nan'))
                         or (-1,) for v, s in zip(basic['video_id'], tags)}

    # ---------------- fit: train window only ----------------
    def fit(self, train_split):
        tr = loader.train_outcomes(self.data_dir)
        assert int(tr['date'].max()) <= 20220421, 'aggregates escaped train window'
        assert len(tr) == len(train_split), 'train_outcomes misaligned with train split'
        assert np.array_equal(tr['user_id'].to_numpy(np.int32), train_split.users), \
            'train_outcomes row order differs from train split; LOO would subtract the wrong row'

        self.dur_edges = np.quantile(train_split.X['duration_ms'].to_numpy(),
                                     np.linspace(0, 1, 11)[1:-1])
        lv = tr['long_view'].ne(0).to_numpy(np.float64)
        ck = tr['is_click'].ne(0).to_numpy(np.float64)
        lk = tr['is_like'].ne(0).to_numpy(np.float64)
        self.g_lv, self.g_click, self.g_like = lv.mean(), ck.mean(), lk.mean()

        uid = tr['user_id'].to_numpy(); vid = tr['video_id'].to_numpy()
        aid = tr['video_id'].map(self.vid2author).fillna(-1).to_numpy()
        one = np.ones(len(tr))

        def agg(keys, vals):
            return pd.Series(vals).groupby(keys).sum()

        self.v_pos, self.v_cnt = agg(vid, lv), agg(vid, one)
        self.v_ckpos = agg(vid, ck)
        self.a_pos, self.a_cnt = agg(aid, lv), agg(aid, one)
        self.u_pos, self.u_cnt = agg(uid, lv), agg(uid, one)
        self.u_ckpos, self.u_lkpos = agg(uid, ck), agg(uid, lk)

        # user x author / user x tag positive counts
        self.ua_pos = pd.Series(lv).groupby([uid, aid]).sum()
        self.u_npos = agg(uid, lv)
        tags = [self.vid2tags.get(v, (-1,)) for v in vid]
        nt = np.fromiter(map(len, tags), np.int32, len(vid))
        ft = np.fromiter((t for ts in tags for t in ts), np.int64, int(nt.sum()))
        fu = np.repeat(uid, nt)
        fl = np.repeat(lv, nt)
        self.ut_pos = pd.Series(fl).groupby([fu, ft]).sum()
        self.tag_prior = (pd.Series(fl).groupby(ft).sum() /
                          max(fl.sum(), 1.0)).astype(np.float32)

        ld = np.log1p(tr['duration_ms'].to_numpy())
        self.u_dsum = agg(uid, ld * lv)
        self.g_dpref = float((ld * lv).sum() / max(lv.sum(), 1))
        self._train_lv, self._train_ld = lv, ld
        self._train_ck, self._train_lk = ck, lk
        self.enc_v_lv = _Encoder(vid, lv)
        self.enc_v_ck = _Encoder(vid, ck)
        self.enc_a_lv = _Encoder(aid, lv)
        self.enc_u_lv = _Encoder(uid, lv)
        self.enc_u_ck = _Encoder(uid, ck)
        self.enc_u_lk = _Encoder(uid, lk)
        self._fit_cf(uid, vid, lv)
        return self

    def _fit_cf(self, uid, vid, lv, k=64, seed=0):
        """Collaborative-filtering item embeddings from the TRAIN window only.

        The organizers flag user behaviour sequences as the one completely
        unexplored direction: the 5 base fields use no history at all. This is
        the cheap CPU stand-in for DIN/SIM-style interest modelling -- factorise
        the user x item long-view matrix, then score a candidate by how well it
        matches what the user actually long-viewed.

        Unlike a popularity aggregate, this varies WITHIN a user's list AND is
        personalised, so it is the one feature family that is not subsumed by
        the video_id categorical.
        """
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import TruncatedSVD
        m = lv > 0
        self.n_u = int(uid.max()) + 1
        self.n_v = int(max(vid.max(), max(self.vid2author) if self.vid2author else 0)) + 1
        R = csr_matrix((np.ones(int(m.sum()), np.float32),
                        (uid[m].astype(np.int64), vid[m].astype(np.int64))),
                       shape=(self.n_u, self.n_v))
        # damp blockbusters so the factors carry preference, not popularity
        pop = np.asarray(R.sum(0)).ravel()
        self.cf_item_norm = 1.0 / np.sqrt(pop + 10.0)
        Rn = R.multiply(self.cf_item_norm[None, :]).tocsr()
        svd = TruncatedSVD(n_components=k, random_state=seed)
        svd.fit(Rn)
        self.cf_V = svd.components_.T.astype(np.float32)      # items x k
        self.cf_profile = (Rn @ self.cf_V).astype(np.float32)  # users x k
        self.cf_k = k
        # out-of-fold profiles: a train row's profile is built from the folds it
        # is NOT in, so there is no self-inclusion and -- unlike leave-one-out --
        # no label-dependent offset for the tree to invert.
        self.cf_folds = np.random.default_rng(seed).integers(0, 5, len(uid))
        self.cf_profile_oof = []
        for f in range(5):
            sel = m & (self.cf_folds != f)
            Rf = csr_matrix((np.ones(int(sel.sum()), np.float32),
                             (uid[sel].astype(np.int64), vid[sel].astype(np.int64))),
                            shape=(self.n_u, self.n_v))
            self.cf_profile_oof.append(
                (Rf.multiply(self.cf_item_norm[None, :]).tocsr() @ self.cf_V
                 ).astype(np.float32))

    # ---------------- transform ----------------
    def build(self, split, blocks, loo=None, encoding='oof'):
        """`loo` defaults to True for the train split. Never for eval splits."""
        if loo is None:
            loo = (split.name == 'train')
        if loo:
            assert split.name == 'train', 'leave-one-out is only valid on train rows'
        X = split.X
        uid = X['user_id'].to_numpy(); vid = X['video_id'].to_numpy()
        aid = X['video_id'].map(self.vid2author).fillna(-1).to_numpy()
        own = self._train_lv if loo else np.zeros(len(X))
        own_ck = (loo and self._train_lv is not None)
        z = np.zeros(len(X))
        P = self.prior
        out = {}

        def look(series, keys, fill=0.0):
            return series.reindex(keys).fillna(fill).to_numpy(np.float64)

        if 'base5' in blocks:
            out['user_id'] = uid
            out['video_id'] = vid
            out['author_id'] = aid.astype(np.int64)
            out['tab'] = X['tab'].to_numpy(np.int16)
            out['dur_bucket'] = np.searchsorted(
                self.dur_edges, X['duration_ms'].to_numpy()).astype(np.int16)

        if 'duration' in blocks:
            out['duration_ms'] = X['duration_ms'].to_numpy(np.float32)

        if 'item_agg' in blocks:
            o_lv = self._train_lv if loo else None
            o_ck = self._train_ck if loo else None
            m = encoding if loo else 'naive'
            out['v_lv_rate'], vc = self.enc_v_lv.encode(vid, P, self.g_lv, m, o_lv)
            out['v_click_rate'], _ = self.enc_v_ck.encode(vid, P, self.g_click, m, o_ck)
            out['a_lv_rate'], _ = self.enc_a_lv.encode(aid, P, self.g_lv, m, o_lv)
            out['v_impressions'] = np.log1p(vc)

        if 'cross_agg' in blocks:
            uap = look(self.ua_pos, pd.MultiIndex.from_arrays([uid, aid]))
            npos = look(self.u_npos, uid)
            out['ua_affinity'] = (uap - own) / (npos - own + 5.0)
            tags = [self.vid2tags.get(v, (-1,)) for v in vid]
            nt = np.fromiter(map(len, tags), np.int32, len(vid))
            ft = np.fromiter((t for ts in tags for t in ts), np.int64, int(nt.sum()))
            fu = np.repeat(uid, nt)
            aff = look(self.ut_pos, pd.MultiIndex.from_arrays([fu, ft]))
            pri = look(self.tag_prior, ft)
            st = np.concatenate([[0], np.cumsum(nt)[:-1]]).astype(np.int64)
            out['ut_affinity'] = ((np.add.reduceat(aff, st) / nt - own) /
                                  (npos - own + 5.0))
            out['ut_tag_prior'] = np.add.reduceat(pri, st) / nt
            ld = np.log1p(X['duration_ms'].to_numpy())
            dsum = look(self.u_dsum, uid)
            pref = np.where(npos - own > 0,
                            (dsum - own * ld) / np.maximum(npos - own, 1e-9),
                            self.g_dpref)
            out['dur_vs_user_pref'] = ld - pref

        if 'cf' in blocks:
            V = self.cf_V
            vi = np.clip(vid.astype(np.int64), 0, self.n_v - 1)
            ui = np.clip(uid.astype(np.int64), 0, self.n_u - 1)
            iv = V[vi]                                   # candidate item factors
            if not loo or encoding == 'naive':
                prof = self.cf_profile[ui].copy()
            elif encoding == 'loo':
                prof = self.cf_profile[ui].copy()
                prof -= (own * self.cf_item_norm[vi])[:, None] * iv
            else:                                   # 'oof'
                prof = np.empty((len(ui), self.cf_k), np.float32)
                for f in range(5):
                    sel = self.cf_folds == f
                    prof[sel] = self.cf_profile_oof[f][ui[sel]]
            dot = np.einsum('ij,ij->i', prof, iv)
            pn = np.linalg.norm(prof, axis=1); vn = np.linalg.norm(iv, axis=1)
            out['cf_dot'] = dot
            out['cf_cos'] = dot / (pn * vn + 1e-6)
            out['cf_profile_norm'] = pn

        if 'dur_feats' in blocks:
            d = X['duration_ms'].to_numpy(np.float32)
            out['log_duration'] = np.log1p(d)
            out['dur_rank_in_list'] = (pd.Series(d).groupby(uid)
                                       .rank(pct=True).to_numpy(np.float32))

        if 'user_agg' in blocks:
            o_lv = self._train_lv if loo else None
            o_ck = self._train_ck if loo else None
            o_lk = self._train_lk if loo else None
            m = encoding if loo else 'naive'
            out['u_lv_rate'], _ = self.enc_u_lv.encode(uid, P, self.g_lv, m, o_lv)
            out['u_click_rate'], _ = self.enc_u_ck.encode(uid, P, self.g_click, m, o_ck)
            out['u_like_rate'], _ = self.enc_u_lk.encode(uid, P, self.g_like, m, o_lk)

        F = pd.DataFrame({k: np.asarray(v, np.float32) if k not in CATEGORICAL
                          else v for k, v in out.items()})
        assert_no_deny_columns(F.columns, f'features.build({split.name!r})')
        assert np.isfinite(F.select_dtypes('float32').to_numpy()).all(), \
            'non-finite feature produced'
        return F
