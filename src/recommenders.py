"""All recommendation algorithms, implemented from the course notes.

Design: every recommender is a class with the SAME interface so the notebook
and the app can treat them interchangeably and a future UI can swap in without
touching this file:

    model = SomeRecommender().fit(train_ratings, movies, tags)
    model.recommend(user_id, n=10)   -> DataFrame[movieId, title, genres, score]
    model.predict_all(user_id)       -> Series of predicted ratings (rating models)

Rating models (UserCF, ItemCF, SVD) also support RMSE/MAE via predict_all.
The non-personalised baselines and the content model only rank, so they leave
predict_all returning NaN.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import build_matrix


# --------------------------------------------------------------------------- #
# Base class: shared bookkeeping (titles, already-seen items, output format)   #
# --------------------------------------------------------------------------- #

class Recommender:
    name = "base"

    def fit(self, ratings, movies, tags=None):
        self.ratings = ratings
        self.movies = movies.set_index("movieId")
        # items each user has already rated -> excluded from recommendations
        self.seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()
        # how many times each movie was rated -> used to drop unreliable tail items
        self.support = ratings.groupby("movieId").size()
        return self

    def predict_all(self, user_id):
        """Predicted ratings for every movie. NaN for ranking-only models."""
        return pd.Series(np.nan, index=self.movies.index)

    def predict(self, user_id, movie_id, scale=(0.5, 5.0)):
        # clip to the valid rating range (the notes: predictions may fall outside it)
        p = float(self.predict_all(user_id).get(movie_id, np.nan))
        return float(np.clip(p, *scale)) if np.isfinite(p) else p

    def _format(self, scores, user_id, n, exclude_seen=True, min_support=None):
        """Turn a Series of scores (indexed by movieId) into a top-n DataFrame.

        `min_support` drops movies with too few ratings to be trustworthy — this
        keeps neighbourhood CF from surfacing tail items backed by a single
        neighbour (the small-overlap problem from the notes).
        """
        scores = scores.dropna()
        if min_support:
            keep = self.support[self.support >= min_support].index
            scores = scores[scores.index.isin(keep)]
        if exclude_seen:
            seen = self.seen.get(user_id, set())
            scores = scores.drop(index=scores.index.intersection(seen))
        top = scores.sort_values(ascending=False).head(n)
        info = self.movies.reindex(top.index)
        return pd.DataFrame(
            {
                "movieId": top.index,
                "title": info["title"].values,
                "genres": info["genres"].values,
                "score": top.values,
            }
        ).reset_index(drop=True)

    def recommend(self, user_id, n=10):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. Popularity baseline (Non-personalised, Top-N by number of ratings)        #
# --------------------------------------------------------------------------- #

class Popularity(Recommender):
    name = "Popularity"

    def fit(self, ratings, movies, tags=None):
        super().fit(ratings, movies, tags)
        self.scores = ratings.groupby("movieId").size().astype(float)
        return self

    def recommend(self, user_id, n=10):
        return self._format(self.scores.copy(), user_id, n)


# --------------------------------------------------------------------------- #
# 2. Bayesian weighted rating (the IMDb formula)                               #
#    WR = v/(v+m)*R + m/(v+m)*C   -- fixes the "one 5-star rating" bias        #
# --------------------------------------------------------------------------- #

class BayesianAverage(Recommender):
    name = "Bayesian Average"

    def fit(self, ratings, movies, tags=None, m=None):
        super().fit(ratings, movies, tags)
        stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
        v, R = stats["count"], stats["mean"]
        C = ratings["rating"].mean()                 # mean rating across catalog
        if m is None:
            m = v.quantile(0.90)                      # min votes to be charted
        self.scores = v / (v + m) * R + m / (v + m) * C
        return self

    def recommend(self, user_id, n=10):
        return self._format(self.scores.copy(), user_id, n)


# --------------------------------------------------------------------------- #
# 3. User-based collaborative filtering                                        #
#    S(u,i) = r̄_u + Σ_v (r_vi - r̄_v) w_uv / Σ_v |w_uv|                        #
#    w_uv = cosine similarity on mean-centred ratings (~ Pearson)              #
# --------------------------------------------------------------------------- #

class UserCF(Recommender):
    name = "User-based CF"

    def fit(self, ratings, movies, tags=None, k=40, min_support=20):
        super().fit(ratings, movies, tags)
        self.k = k
        self.min_support = min_support
        M = build_matrix(ratings)                    # users x items, NaN missing
        self.M = M
        self.user_mean = M.mean(axis=1)              # r̄_u
        self.centered = M.sub(self.user_mean, axis=0).fillna(0.0)
        self.rated_mask = M.notna()                  # who actually rated what
        sim = cosine_similarity(self.centered.values)
        np.fill_diagonal(sim, 0.0)                    # a user is not its own neighbour
        self.sim = pd.DataFrame(sim, index=M.index, columns=M.index)
        return self

    def predict_all(self, user_id):
        if user_id not in self.M.index:
            return pd.Series(np.nan, index=self.M.columns)
        neighbours = self.sim.loc[user_id].nlargest(self.k)   # top-k similar users
        w = neighbours.values                                  # (k,)
        dev = self.centered.loc[neighbours.index].values       # (k, items)
        rated = self.rated_mask.loc[neighbours.index].values   # (k, items)
        num = (w[:, None] * dev).sum(axis=0)
        den = (np.abs(w)[:, None] * rated).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pred = self.user_mean[user_id] + np.where(den > 0, num / den, np.nan)
        pred = np.where(np.isfinite(pred), np.clip(pred, 0.5, 5.0), pred)
        return pd.Series(pred, index=self.M.columns)

    def recommend(self, user_id, n=10):
        return self._format(self.predict_all(user_id), user_id, n,
                            min_support=self.min_support)


# --------------------------------------------------------------------------- #
# 4. Item-based collaborative filtering (adjusted cosine)                      #
#    similarity between items on user-mean-centred ratings;                    #
#    S(u,i) = Σ_j r_uj * w_ij / Σ_j |w_ij|  over items j the user rated        #
# --------------------------------------------------------------------------- #

class ItemCF(Recommender):
    name = "Item-based CF"

    def fit(self, ratings, movies, tags=None, min_support=20):
        super().fit(ratings, movies, tags)
        self.min_support = min_support
        M = build_matrix(ratings)
        self.M = M
        self.items = M.columns                                 # movieId per matrix column
        self.item_pos = {mid: i for i, mid in enumerate(self.items)}
        user_mean = M.mean(axis=1)
        centered = M.sub(user_mean, axis=0).fillna(0.0)        # adjusted cosine
        # Sparse + float32: the item-item similarity matrix is mostly zeros (most
        # movie pairs share no rater), so we store only the non-zeros. This keeps
        # ~9.7k x 9.7k from blowing past a 512 MB box (dense float64 ≈ 760 MB).
        C = csr_matrix(centered.T.values.astype(np.float32))   # items x users
        norms = np.sqrt(C.multiply(C).sum(axis=1)).A.ravel()   # L2 norm per item
        norms[norms == 0] = 1.0
        Cn = diags(1.0 / norms) @ C                            # unit-length rows
        sim = (Cn @ Cn.T).tocsr()                              # cosine sims, sparse
        sim.setdiag(0.0)                                       # not its own neighbour
        sim.data = np.clip(sim.data, 0.0, None)                # drop negative correlations
        sim.eliminate_zeros()
        self.sim = sim.astype(np.float32)
        return self

    def predict_all(self, user_id):
        if user_id not in self.M.index:
            return pd.Series(np.nan, index=self.items)
        rated = self.M.loc[user_id].dropna()
        if rated.empty:
            return pd.Series(np.nan, index=self.items)
        rows = [self.item_pos[m] for m in rated.index]
        W = self.sim[rows]                           # (rated, all items), >= 0
        r = rated.values.astype(np.float32)          # (rated,)
        num = np.asarray(W.T @ r).ravel()            # Σ_j r_uj * w_ij
        den = np.asarray(W.sum(axis=0)).ravel()      # Σ_j |w_ij|
        with np.errstate(invalid="ignore", divide="ignore"):
            pred = np.where(den > 0, num / den, np.nan)
        return pd.Series(pred, index=self.items)

    def recommend(self, user_id, n=10):
        return self._format(self.predict_all(user_id), user_id, n,
                            min_support=self.min_support)


# --------------------------------------------------------------------------- #
# 5. Content-based filtering (TF-IDF over genres + tags, cosine to a profile)  #
# --------------------------------------------------------------------------- #

class ContentBased(Recommender):
    name = "Content-based"

    def fit(self, ratings, movies, tags=None):
        super().fit(ratings, movies, tags)
        m = movies.copy()
        m["genres_text"] = m["genres"].str.replace("|", " ", regex=False)
        if tags is not None and len(tags):
            tag_text = tags.groupby("movieId")["tag"].apply(
                lambda s: " ".join(s.astype(str))
            )
            m = m.merge(tag_text.rename("tag_text"), on="movieId", how="left")
        else:
            m["tag_text"] = ""
        # one "soup" of features per movie -> TF-IDF down-weights common genres
        soup = (m["genres_text"].fillna("") + " " + m["tag_text"].fillna("")).str.lower()
        self.vec = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.item_vecs = self.vec.fit_transform(soup)          # sparse (movies, terms)
        self.movie_ids = m["movieId"].values
        self.row_of = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.user_mean = ratings.groupby("userId")["rating"].mean()
        return self

    def _profile(self, user_id, user_ratings=None):
        """Build the user's taste vector = like/dislike-weighted item vectors."""
        ur = user_ratings
        if ur is None:
            ur = self.ratings[self.ratings.userId == user_id]
        ur = ur[ur.movieId.isin(self.row_of)]
        if ur.empty:
            return None
        mean = float(ur.rating.mean())
        idx = [self.row_of[m] for m in ur.movieId]
        weights = ur.rating.values - mean                      # +liked / -disliked
        profile = self.item_vecs[idx].multiply(weights[:, None]).sum(axis=0)
        return np.asarray(profile)                             # (1, terms)

    def predict_all(self, user_id):
        # content model scores are cosine similarities, not 1-5 ratings
        return pd.Series(np.nan, index=self.movies.index)

    def recommend(self, user_id, n=10, user_ratings=None):
        profile = self._profile(user_id, user_ratings)
        if profile is None:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])
        sims = cosine_similarity(profile, self.item_vecs).ravel()
        scores = pd.Series(sims, index=self.movie_ids)
        return self._format(scores, user_id, n)

    def explain(self, user_id, movie_id, top=3):
        """Top shared feature words between the movie and the user's profile."""
        profile = self._profile(user_id)
        if profile is None or movie_id not in self.row_of:
            return []
        terms = np.asarray(self.vec.get_feature_names_out())
        item_row = self.item_vecs[self.row_of[movie_id]].toarray().ravel()
        shared = (profile.ravel() > 0) & (item_row > 0)
        if not shared.any():
            return []
        order = np.argsort(profile.ravel() * shared)[::-1]
        return [t for t in terms[order][:top]]


# --------------------------------------------------------------------------- #
# 6. Matrix factorisation via Truncated SVD (latent factors)                   #
#    Decompose the mean-centred matrix, reconstruct, add the user mean back.   #
# --------------------------------------------------------------------------- #

class SVDRecommender(Recommender):
    name = "Matrix Factorisation (SVD)"

    def fit(self, ratings, movies, tags=None, k=50):
        super().fit(ratings, movies, tags)
        M = build_matrix(ratings)
        self.M = M
        self.user_mean = M.mean(axis=1)
        centered = M.sub(self.user_mean, axis=0).fillna(0.0)
        k = min(k, min(centered.shape) - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        U = self.svd.fit_transform(csr_matrix(centered.values))     # users x k
        recon = U @ self.svd.components_                            # users x items
        self.recon = pd.DataFrame(recon, index=M.index, columns=M.columns)
        return self

    def predict_all(self, user_id):
        if user_id not in self.recon.index:
            return pd.Series(np.nan, index=self.M.columns)
        return self.recon.loc[user_id] + self.user_mean[user_id]

    def recommend(self, user_id, n=10):
        return self._format(self.predict_all(user_id), user_id, n)


# --------------------------------------------------------------------------- #
# Registry — single source of truth for the notebook and the app              #
# --------------------------------------------------------------------------- #

RECOMMENDERS = [
    Popularity,
    BayesianAverage,
    UserCF,
    ItemCF,
    ContentBased,
    SVDRecommender,
]


def all_recommenders():
    """name -> class, in presentation order."""
    return {cls.name: cls for cls in RECOMMENDERS}
