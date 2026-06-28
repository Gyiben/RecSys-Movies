"""Dataset summary + EDA chart data as plain JSON for the Insights tab.

Mirrors ``notebooks/01_eda.ipynb``. Cheap to compute (simple groupbys), so no
caching is needed.
"""

from .data import load_data


def get_eda():
    ratings, movies, tags = load_data()
    n_users = int(ratings.userId.nunique())
    n_movies = int(ratings.movieId.nunique())
    n_ratings = int(len(ratings))
    sparsity = 1 - n_ratings / (n_users * n_movies)

    rating_dist = ratings.rating.value_counts().sort_index()

    counts = ratings.groupby("movieId").size().sort_values(ascending=False).values
    half_movies = int((counts.cumsum() <= 0.5 * counts.sum()).sum())

    genres = movies.genres.str.get_dummies("|").sum().sort_values(ascending=False)
    genres = genres[genres.index != "(no genres listed)"].head(12)

    rpu = ratings.groupby("userId").size()
    rpm = ratings.groupby("movieId").size()

    # long-tail curve, downsampled so the chart stays light
    step = max(1, len(counts) // 200)
    tail = [int(c) for c in counts[::step]]

    return {
        "summary": {
            "users": n_users, "movies": n_movies, "ratings": n_ratings,
            "sparsity": sparsity, "meanRating": round(float(ratings.rating.mean()), 3),
            "tags": int(len(tags)),
            "medianPerUser": int(rpu.median()), "medianPerMovie": int(rpm.median()),
            "halfRatingsMovies": half_movies,
            "halfRatingsPct": half_movies / len(counts),
        },
        "ratingDist": {
            "labels": [float(x) for x in rating_dist.index],
            "values": [int(x) for x in rating_dist.values],
        },
        "genres": {
            "labels": list(genres.index),
            "values": [int(x) for x in genres.values],
        },
        "longTail": tail,
    }
