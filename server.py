"""MovieLens recommender — FastAPI backend.

A thin layer over ``src/recommenders.py``: it collects input, calls a model's
``.recommend(...)``, enriches the result with poster artwork, and serves a
single-page dark UI from ``static/index.html``. No algorithm lives here.

Run:  uv run uvicorn server:app --reload
Then: http://127.0.0.1:8000
"""

import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data import load_data
from src.eda import get_eda
from src.evaluation import get_evaluation
from src.posters import PosterService
from src.recommenders import all_recommenders

# What the "score" column means for each family of model (shown to the user).
SCORE_HELP = {
    "Popularity": "ratings",
    "Bayesian Average": "weighted ★",
    "User-based CF": "predicted ★",
    "Item-based CF": "predicted ★",
    "Content-based": "match",
    "Matrix Factorisation (SVD)": "predicted ★",
}

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Movie Recommender")

ratings, movies, tags = load_data()
posters = PosterService()
_models = {}  # method name -> fitted model (existing-user mode, full data)

# Warm the (slow) evaluation cache in the background so the Insights tab is ready
# by the time anyone opens it. No-op once .eval_cache.json exists.
threading.Thread(target=get_evaluation, daemon=True).start()


def get_model(method):
    """Fit and cache a model on the full dataset (existing-user mode)."""
    if method not in _models:
        _models[method] = all_recommenders()[method]().fit(ratings, movies, tags)
    return _models[method]


def enrich(recs, method, model=None, user_id=None):
    """Turn a recommendations DataFrame into a JSON-ready list with posters."""
    if recs is None or recs.empty:
        return []
    pmap = posters.posters_for(recs.movieId.tolist())
    out = []
    for _, row in recs.iterrows():
        mid = int(row.movieId)
        item = {
            "movieId": mid,
            "title": row.title,
            "genres": [g for g in row.genres.split("|") if g != "(no genres listed)"],
            "score": round(float(row.score), 2),
            "scoreLabel": SCORE_HELP[method],
            "poster": pmap.get(mid),
            "why": [],
        }
        if method == "Content-based" and model is not None and user_id is not None:
            item["why"] = model.explain(user_id, mid) or []
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# API                                                                         #
# --------------------------------------------------------------------------- #

class RecRequest(BaseModel):
    method: str
    mode: str                       # "existing" | "coldstart"
    n: int = 10
    user_id: int | None = None
    new_ratings: list[dict] | None = None   # [{movieId, rating}, ...]


@app.get("/api/meta")
def meta():
    return {
        "methods": list(all_recommenders()),
        "scoreHelp": SCORE_HELP,
        "users": sorted(int(u) for u in ratings.userId.unique()),
        "postersEnabled": posters.enabled,
    }


@app.get("/api/profile/{user_id}")
def profile(user_id: int):
    liked = (ratings[ratings.userId == user_id]
             .merge(movies, on="movieId").nlargest(8, "rating"))
    pmap = posters.posters_for(liked.movieId.tolist())
    return [
        {"movieId": int(r.movieId), "title": r.title,
         "genres": [g for g in r.genres.split("|") if g != "(no genres listed)"],
         "rating": float(r.rating), "poster": pmap.get(int(r.movieId))}
        for _, r in liked.iterrows()
    ]


@app.get("/api/eda")
def eda():
    """Dataset summary + chart data for the Insights tab (cheap)."""
    return get_eda()


@app.get("/api/metrics")
def metrics_endpoint():
    """Cached offline evaluation of all six models (slow first run, then instant)."""
    return get_evaluation()


@app.get("/api/popular")
def popular(limit: int = 40):
    pop = ratings.groupby("movieId").size().nlargest(limit).index
    sel = movies[movies.movieId.isin(pop)]
    pmap = posters.posters_for(sel.movieId.tolist())
    return [
        {"movieId": int(r.movieId), "title": r.title,
         "genres": [g for g in r.genres.split("|") if g != "(no genres listed)"],
         "poster": pmap.get(int(r.movieId))}
        for _, r in sel.iterrows()
    ]


@app.post("/api/recommend")
def recommend(req: RecRequest):
    if req.mode == "existing":
        if req.user_id is None:
            return []
        model = get_model(req.method)
        recs = model.recommend(req.user_id, n=req.n)
        return enrich(recs, req.method, model, req.user_id)

    # cold start: append a brand-new user's ratings and fit fresh (uncached)
    nr = pd.DataFrame(req.new_ratings or [])
    if nr.empty:
        return []
    new_uid = int(ratings.userId.max()) + 1
    rows = pd.DataFrame(
        {"userId": new_uid, "movieId": nr.movieId,
         "rating": nr.rating, "timestamp": int(time.time())}
    )
    full = pd.concat([ratings, rows], ignore_index=True)
    model = all_recommenders()[req.method]().fit(full, movies, tags)
    if req.method == "Content-based":   # profile comes straight from your ratings
        recs = model.recommend(new_uid, n=req.n,
                               user_ratings=nr.assign(userId=new_uid))
    else:
        recs = model.recommend(new_uid, n=req.n)
    return enrich(recs, req.method, model, new_uid)


# --------------------------------------------------------------------------- #
# Static front-end                                                            #
# --------------------------------------------------------------------------- #

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
