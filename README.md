# Movie Recommender — Individual Project

Movie-track recommender prototype for the ESADE Recommender Systems course. Six algorithms on [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/), compared with ranking and beyond-accuracy metrics, plus a FastAPI web app with a dark cinematic UI: a **Discover** tab (hero + carousels) and an **Insights** tab (EDA charts, a trade-off radar, and the full evaluation table).

## Dataset

**Source:** GroupLens MovieLens Latest Small (`ml-latest-small/`)

- 610 users, 9,742 movies, 100,836 ratings (0.5–5 stars), plus user tags
- Downloaded from [grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)
- License: research/non-commercial use; acknowledge GroupLens in publications (see `ml-latest-small/README.txt`)

**Preprocessing:**

- Temporal train/test split: each user's most recent 20% of ratings held out (`src/data.py`)
- User/item CF: candidates need ≥20 ratings (MovieLens convention) to filter obscure tail items

## Methods

| # | Algorithm | Module |
|---|-----------|--------|
| 1 | Popularity (Top-N by rating count) | `Popularity` |
| 2 | Bayesian average (IMDb formula) | `BayesianAverage` |
| 3 | User-based collaborative filtering | `UserCF` |
| 4 | Item-based collaborative filtering | `ItemCF` |
| 5 | Content-based (TF-IDF on genres + tags) | `ContentBased` |
| 6 | Matrix factorisation (Truncated SVD) | `SVDRecommender` |

All models share one interface: `model.fit(ratings, movies, tags)` then `model.recommend(user_id, n=10)`.

## Setup

Requires Python ≥3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

**Web app** (FastAPI backend + single-page UI, fits on the full dataset):

```bash
uv run uvicorn server:app --reload
```

Then open http://127.0.0.1:8000. For real movie posters, set a free
[TMDb](https://www.themoviedb.org/settings/api) v3 key first; without it the UI
falls back to styled text tiles:

```bash
# PowerShell:  $env:TMDB_API_KEY = "your_key"
export TMDB_API_KEY=your_key
```

**Notebooks:**

- `notebooks/01_eda.ipynb` — exploratory data analysis
- `notebooks/02_evaluation.ipynb` — temporal-split evaluation of all six methods

Evaluation fits on train only; the app uses all ratings so professors can try any user without re-running the notebook.

## Project layout

```
server.py               FastAPI backend (thin layer over recommenders)
static/index.html       single-page dark cinematic UI
render.yaml             Render deployment blueprint
src/data.py             load MovieLens, build matrix, temporal split
src/recommenders.py     six algorithms (used by app + evaluation)
src/recommenders_simple.py  readable versions for study
src/posters.py          TMDb poster lookup (cached) for the UI
src/metrics.py          RMSE, NDCG, coverage, novelty, diversity, …
src/evaluation.py       cached temporal-split evaluation of all models (Insights tab)
src/eda.py              dataset summary + chart data (Insights tab)
ml-latest-small/        MovieLens dataset (bundled)
notebooks/              EDA and evaluation
```
