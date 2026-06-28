"""Offline evaluation of every recommender on a temporal split.

A reusable port of ``notebooks/02_evaluation.ipynb`` so the web app can show the
same numbers. Fitting six models and ranking for every test user is slow, so the
result is cached to disk and only recomputed when missing (or refresh=True).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from . import metrics as mt
from .data import load_data, temporal_split
from .recommenders import ContentBased, all_recommenders

CACHE_FILE = Path(__file__).resolve().parents[1] / ".eval_cache.json"
K = 10


def _evaluate_all():
    ratings, movies, tags = load_data()
    train, test = temporal_split(ratings, test_frac=0.2)

    # a held-out item counts as relevant if the user rated it >= 4 in the future
    relevant = (test[test.rating >= 4]
                .groupby("userId")["movieId"].apply(list).to_dict())
    test_users = sorted(test.userId.unique())
    popularity = train.groupby("movieId").size().to_dict()
    catalog = train.movieId.unique()

    # content vectors (sparse) reused for intra-list diversity
    cb = ContentBased().fit(train, movies, tags)

    def diversity(reclist):
        rows = [cb.row_of[m] for m in reclist if m in cb.row_of]
        if len(rows) < 2:
            return 0.0
        S = cosine_similarity(cb.item_vecs[rows])
        iu = np.triu_indices(len(rows), 1)
        return float(1 - S[iu].mean())

    def evaluate(model):
        model.fit(train, movies, tags)
        is_rating = bool(model.predict_all(test_users[0]).notna().any())
        P = R = N = M = 0.0
        rec_all, p_true, p_pred = [], [], []
        for u in test_users:
            rl = list(model.recommend(u, n=K).movieId)
            rel = relevant.get(u, [])
            P += mt.precision_at_k(rl, rel, K)
            R += mt.recall_at_k(rl, rel, K)
            N += mt.ndcg_at_k(rl, rel, K)
            M += mt.mrr(rl, rel)
            rec_all.append(rl)
            if is_rating:
                preds = model.predict_all(u)
                ut = test[test.userId == u]
                for mid, tr in zip(ut.movieId, ut.rating):
                    pv = preds.get(mid, np.nan)
                    if np.isfinite(pv):
                        p_pred.append(float(pv))
                        p_true.append(float(tr))
        nu = len(test_users)
        flat = [m for rl in rec_all for m in rl]
        return {
            "Precision@10": P / nu, "Recall@10": R / nu,
            "NDCG@10": N / nu, "MRR": M / nu,
            "Coverage": mt.coverage(flat, catalog),
            "Novelty": mt.novelty(rec_all, popularity),
            "Diversity": float(np.mean([diversity(rl) for rl in rec_all])),
            "RMSE": mt.rmse(p_pred, p_true) if p_pred else None,
            "MAE": mt.mae(p_pred, p_true) if p_pred else None,
        }

    results = {name: evaluate(cls()) for name, cls in all_recommenders().items()}
    meta = {"train": int(len(train)), "test": int(len(test)),
            "testUsers": int(len(test_users)), "k": K}
    return {"results": results, "meta": meta}


def get_evaluation(refresh=False):
    """Cached evaluation for all models. Computes once, reads from disk after."""
    if not refresh and CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    data = _evaluate_all()
    try:
        CACHE_FILE.write_text(json.dumps(data))
    except Exception:
        pass
    return data
