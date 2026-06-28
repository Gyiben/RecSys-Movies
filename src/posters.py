"""Poster artwork lookup for the web UI.

Maps a MovieLens ``movieId`` -> TMDb poster image URL using ``links.csv``
(which already carries each film's ``tmdbId``). Results are cached on disk so
we hit the TMDb API at most once per movie across runs.

Set a free TMDb v3 API key in the ``TMDB_API_KEY`` environment variable to turn
real posters on. With no key the service stays disabled and the UI falls back
to styled text tiles, so everything still works offline.
"""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "ml-latest-small"
CACHE_FILE = Path(__file__).resolve().parents[1] / ".poster_cache.json"
IMG_BASE = "https://image.tmdb.org/t/p/w342"
API = "https://api.themoviedb.org/3/movie/{tmdb}?api_key={key}"


class PosterService:
    """Lazy, cached TMDb poster lookups keyed by MovieLens movieId."""

    def __init__(self):
        self.key = os.environ.get("TMDB_API_KEY", "").strip()
        links = pd.read_csv(DATA_DIR / "links.csv")
        self.tmdb_of = dict(zip(links.movieId, links.tmdbId))
        self.cache = {}
        if CACHE_FILE.exists():
            try:
                self.cache = json.loads(CACHE_FILE.read_text())
            except Exception:
                self.cache = {}

    @property
    def enabled(self):
        return bool(self.key)

    def _fetch_one(self, movie_id):
        """Resolve and cache the poster URL for a single movie (None if absent)."""
        tmdb = self.tmdb_of.get(movie_id)
        url = None
        if pd.notna(tmdb):
            try:
                req = API.format(tmdb=int(tmdb), key=self.key)
                with urllib.request.urlopen(req, timeout=6) as r:
                    data = json.loads(r.read())
                pp = data.get("poster_path")
                if pp:
                    url = IMG_BASE + pp
            except Exception:
                url = None
        self.cache[str(movie_id)] = url
        return url

    def posters_for(self, movie_ids):
        """movieId -> poster URL (or None). Fetches missing ones concurrently."""
        ids = [int(m) for m in movie_ids]
        if not self.key:
            return {m: None for m in ids}
        todo = [m for m in ids if str(m) not in self.cache]
        if todo:
            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(self._fetch_one, todo))
            self._save()
        return {m: self.cache.get(str(m)) for m in ids}

    def _save(self):
        try:
            CACHE_FILE.write_text(json.dumps(self.cache))
        except Exception:
            pass
