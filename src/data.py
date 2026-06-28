"""Data loading and train/test splitting for the MovieLens recommender.

Everything downstream (algorithms, metrics, app) reuses these three helpers so
the data handling lives in exactly one place.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "ml-latest-small"


def load_data():
    """Return (ratings, movies, tags) as DataFrames."""
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    return ratings, movies, tags


def build_matrix(ratings):
    """User x item ratings matrix (NaN where a user has not rated a movie)."""
    return ratings.pivot_table(index="userId", columns="movieId", values="rating")


def temporal_split(ratings, test_frac=0.2, min_train=5):
    """Hold out each user's most recent ratings as the test set.

    A *temporal* split (train on the past, test on the future) avoids the data
    leakage you get from a random split — see the Evaluation notes, pitfall #2.
    Users with too few ratings keep all of them in train (nothing to test on).
    """
    ratings = ratings.sort_values("timestamp")
    train_parts, test_parts = [], []
    for _, grp in ratings.groupby("userId", sort=False):
        n = len(grp)
        n_test = int(round(n * test_frac))
        if n - n_test < min_train:          # keep at least min_train in training
            n_test = max(0, n - min_train)
        if n_test == 0:
            train_parts.append(grp)
        else:
            train_parts.append(grp.iloc[:-n_test])
            test_parts.append(grp.iloc[-n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = (pd.concat(test_parts) if test_parts else ratings.iloc[0:0]).reset_index(drop=True)
    return train, test
