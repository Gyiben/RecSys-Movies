"""Evaluation metrics, implemented straight from the Evaluation lecture notes.

Two families:
  - rating accuracy (rmse, mae) for models that predict a 1-5 rating,
  - top-N ranking + beyond-accuracy for the recommendation lists themselves.

All functions are pure: they take plain lists/arrays so they are easy to test
and reuse from the evaluation notebook.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Rating-accuracy metrics                                                      #
# --------------------------------------------------------------------------- #

def rmse(preds, truths):
    """Root Mean Squared Error — penalises big misses hard."""
    preds, truths = np.asarray(preds, float), np.asarray(truths, float)
    return float(np.sqrt(np.mean((preds - truths) ** 2)))


def mae(preds, truths):
    """Mean Absolute Error — average rating points off."""
    preds, truths = np.asarray(preds, float), np.asarray(truths, float)
    return float(np.mean(np.abs(preds - truths)))


# --------------------------------------------------------------------------- #
# Top-N ranking metrics (recommended = ordered list of movieIds)              #
# --------------------------------------------------------------------------- #

def precision_at_k(recommended, relevant, k=10):
    """Fraction of the top-k recommendations that are relevant."""
    relevant = set(relevant)
    hits = sum(1 for m in recommended[:k] if m in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    """Fraction of the user's relevant items captured in the top-k."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for m in recommended[:k] if m in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    """Normalised Discounted Cumulative Gain — position-aware quality in [0, 1]."""
    relevant = set(relevant)
    dcg = sum(1 / np.log2(i + 2) for i, m in enumerate(recommended[:k]) if m in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended, relevant):
    """Mean Reciprocal Rank — 1 / rank of the first relevant hit."""
    relevant = set(relevant)
    for i, m in enumerate(recommended):
        if m in relevant:
            return 1 / (i + 1)
    return 0.0


# --------------------------------------------------------------------------- #
# Beyond-accuracy metrics (operate over ALL users' recommendation lists)      #
# --------------------------------------------------------------------------- #

def coverage(all_recommended_items, catalog):
    """Share of the catalog the system is actually able to recommend."""
    return len(set(all_recommended_items)) / len(set(catalog))


def novelty(recommended_lists, popularity):
    """Mean self-information -log2(p(item)): higher = more niche/novel."""
    total = sum(popularity.values())
    scores = [
        -np.log2(popularity.get(m, 1) / total)
        for rec in recommended_lists
        for m in rec
    ]
    return float(np.mean(scores)) if scores else 0.0


def intra_list_diversity(recommended, vectors):
    """1 - average pairwise cosine similarity inside one recommendation list.

    `vectors` maps movieId -> feature vector (we use the content TF-IDF vectors).
    Higher = the list mixes different kinds of items rather than near-duplicates.
    """
    items = [m for m in recommended if m in vectors]
    if len(items) < 2:
        return 0.0
    sims = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            va, vb = vectors[items[a]], vectors[items[b]]
            denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
            sims.append(float(np.dot(va, vb) / denom))
    return 1 - np.mean(sims)
