"""Build step: precompute every existing-user recommendation to JSON.

Run locally whenever the dataset or algorithms change:

    uv run python precompute.py

This fits all six models once (memory-heavy, but offline) and writes a compact
``precomputed_recs.json`` mapping method -> userId -> top-N [{movieId, score,
why?}]. The web server then serves existing-user recommendations straight from
this file, so it never has to fit the heavy CF/SVD models at runtime — that is
what keeps it under the 512 MB hosting limit.
"""

import json
from pathlib import Path

from src.data import load_data
from src.recommenders import all_recommenders

N = 20  # precompute the max the UI slider allows; it slices to fewer
OUT = Path(__file__).resolve().parent / "precomputed_recs.json"


def main():
    ratings, movies, tags = load_data()
    users = sorted(int(u) for u in ratings.userId.unique())
    out = {}

    for name, cls in all_recommenders().items():
        print(f"fitting {name} …", flush=True)
        model = cls().fit(ratings, movies, tags)
        is_content = name == "Content-based"
        per_user = {}
        for uid in users:
            recs = model.recommend(uid, n=N)
            items = []
            for _, row in recs.iterrows():
                mid = int(row.movieId)
                d = {"movieId": mid, "score": round(float(row.score), 2)}
                if is_content:
                    why = model.explain(uid, mid)
                    if why:
                        d["why"] = why
                items.append(d)
            per_user[str(uid)] = items
        out[name] = per_user
        del model  # free before the next (offline, but keep it tidy)

    OUT.write_text(json.dumps({"n": N, "recs": out}))
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.name}: {len(out)} methods x {len(users)} users  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
