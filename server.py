"""MovieLens recommender — FastAPI backend.

A thin layer over ``src/recommenders.py``: it collects input, calls a model's
``.recommend(...)``, enriches the result with poster artwork, and serves a
single-page dark UI from ``static/index.html``. No algorithm lives here.

Run:  uv run uvicorn server:app --reload
Then: http://127.0.0.1:8000
"""

import json
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

# Existing-user recommendations are precomputed offline (precompute.py) so the
# server never fits the heavy CF/SVD models at runtime — that is what keeps it
# well under the 512 MB hosting limit. Cold start is the one case that must run
# live, and it only ever uses the lightweight Content-based model.
PRECOMP = json.loads((BASE / "precomputed_recs.json").read_text())["recs"]
MOVIE_INFO = {
    int(r.movieId): {
        "title": r.title,
        "genres": [g for g in r.genres.split("|") if g != "(no genres listed)"],
    }
    for r in movies.itertuples()
}

_content_model = None


def content_model():
    """Lazily fit (once) the only model the server needs to run live."""
    global _content_model
    if _content_model is None:
        from src.recommenders import ContentBased
        _content_model = ContentBased().fit(ratings, movies, tags)
    return _content_model


def enrich(items, method, user_id=None):
    """items: [{movieId, score, why?}] -> JSON-ready cards with title + poster."""
    if not items:
        return []
    pmap = posters.posters_for([it["movieId"] for it in items])
    out = []
    for it in items:
        mid = it["movieId"]
        info = MOVIE_INFO.get(mid, {"title": str(mid), "genres": []})
        out.append({
            "movieId": mid,
            "title": info["title"],
            "genres": info["genres"],
            "score": round(float(it["score"]), 2),
            "scoreLabel": SCORE_HELP[method],
            "poster": pmap.get(mid),
            "why": it.get("why", []),
        })
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
        method = req.method if req.method in PRECOMP else next(iter(PRECOMP))
        items = PRECOMP.get(method, {}).get(str(req.user_id), [])[:req.n]
        return enrich(items, method, req.user_id)

    # cold start: a brand-new user's ratings -> content-based profile, computed
    # live. CF/SVD cold start would need a full refit the 512 MB box can't
    # afford, so cold start always uses content-based filtering.
    nr = pd.DataFrame(req.new_ratings or [])
    if nr.empty:
        return []
    new_uid = int(ratings.userId.max()) + 1
    ur = nr.assign(userId=new_uid)
    model = content_model()
    recs = model.recommend(new_uid, n=req.n, user_ratings=ur)
    items = []
    for _, row in recs.iterrows():
        mid = int(row.movieId)
        d = {"movieId": mid, "score": round(float(row.score), 2)}
        why = model.explain(new_uid, mid, user_ratings=ur)
        if why:
            d["why"] = why
        items.append(d)
    return enrich(items, "Content-based", new_uid)


# --------------------------------------------------------------------------- #
# Static front-end                                                            #
# --------------------------------------------------------------------------- #

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
