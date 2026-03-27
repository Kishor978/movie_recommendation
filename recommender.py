"""
recommender.py — MODE 2: Load pre-built artifacts and serve queries instantly.

Query API
─────────
All queries go through a single method:

    rec.recommend(title=..., description=..., genre=..., top_n=5)

Pass any one, two, or all three parameters:

    rec.recommend(title="Inception")
    rec.recommend(description="a cop chasing a serial killer")
    rec.recommend(genre="Horror")
    rec.recommend(title="Inception",   genre="Scifi")
    rec.recommend(description="heist gone wrong", genre="Thriller")
    rec.recommend(title="Inception",   description="dream heist")
    rec.recommend(title="Inception",   description="dream heist", genre="Scifi")

Signal combination strategy
────────────────────────────
When multiple signals are provided their score vectors are averaged before ranking:

  title only          -> cached sim row  (O(1) after first call)
  description only    -> one dot product
  title + description -> average of both score vectors (equal weight)
  genre only          -> skips cosine entirely, ranks by IMDB rating
  any + genre         -> cosine scoring first, genre applied as hard filter after
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.preprocessing import normalize

from preprocess import clean_text


DEFAULT_MODEL_DIR = "model"


class MovieRecommender:
    """Inference-only recommender. Loads artifacts from disk once, serves forever."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self._load_artifacts(model_dir)
        self._title_idx: dict[str, int] = {
            t.lower(): i for i, t in enumerate(self.df["title"])
        }
        self._sim_cache: dict[int, np.ndarray] = {}
        # pre-built genre index: genre_lower -> set of row indices
        self._genre_idx: dict[str, set] = {}
        for i, g in enumerate(self.df["genre"]):
            self._genre_idx.setdefault(str(g).lower(), set()).add(i)

    # ──────────────────────────────────────────────────────────────────────────
    # Artifact loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_artifacts(self, model_dir: str):
        vec_path = os.path.join(model_dir, "vectorizer.joblib")
        mat_path = os.path.join(model_dir, "tfidf_matrix.npz")
        df_path  = os.path.join(model_dir, "movies.parquet")
        if not os.path.exists(df_path):
            df_path = os.path.join(model_dir, "movies.csv.gz")

        for p in (vec_path, mat_path, df_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Artifact not found: {p}\n"
                    f"Run  python3 build.py  first."
                )

        t0 = time.time()
        print(f"[load] Reading artifacts from '{model_dir}/' ...")
        self.vectorizer        = joblib.load(vec_path)
        self.tfidf_matrix_norm = sp.load_npz(mat_path)
        self.df = (pd.read_parquet(df_path) if df_path.endswith(".parquet")
                   else pd.read_csv(df_path, index_col=0, compression="gzip"))
        self.df = self.df.reset_index(drop=True)
        print(f"[load] {len(self.df):,} movies | "
              f"matrix {self.tfidf_matrix_norm.shape} | "
              f"loaded in {time.time()-t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_sim_row(self, idx: int) -> np.ndarray:
        """Cosine scores for movie[idx] vs every movie. Cached after first call."""
        if idx not in self._sim_cache:
            row = self.tfidf_matrix_norm[idx]
            self._sim_cache[idx] = (row @ self.tfidf_matrix_norm.T).toarray().flatten()
        return self._sim_cache[idx]

    def _desc_scores(self, description: str) -> np.ndarray:
        """Transform free-text query and dot-product against the whole matrix."""
        cleaned = clean_text(description)
        vec     = self.vectorizer.transform([cleaned])
        vec     = normalize(vec, norm="l2")
        return (vec @ self.tfidf_matrix_norm.T).toarray().flatten()

    def _apply_filters(self, scores: np.ndarray,
                       exclude_idx: int = None,
                       genre: str = None) -> list:
        """Sorted (idx, score) pairs with optional genre filter and self-exclusion."""
        pairs = list(enumerate(scores))
        if exclude_idx is not None:
            pairs = [(i, s) for i, s in pairs if i != exclude_idx]
        if genre:
            g = genre.lower()
            allowed = self._genre_idx.get(g, set())
            pairs = [(i, s) for i, s in pairs if i in allowed]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def _format(self, pairs: list, top_n: int) -> list[dict]:
        results = []
        for idx, score in pairs[:top_n]:
            row = self.df.loc[idx]
            desc = row["description"]
            results.append({
                "title":       row["title"],
                "year":        row.get("year", ""),
                "genre":       row["genre"],
                "rating":      row["rating"] if pd.notna(row["rating"]) else None,
                "similarity":  round(float(score), 4),
                "description": desc[:150] + "..." if len(desc) > 150 else desc,
            })
        return results

    def _resolve_title(self, title: str):
        """Return (idx, None) on success or (None, error_msg) on failure."""
        idx = self._title_idx.get(title.lower())
        if idx is not None:
            return idx, None
        matches = [t for t in self._title_idx if title.lower() in t]
        if matches:
            return None, f"'{title}' not found. Did you mean: {matches[:3]} ?"
        return None, f"'{title}' not found in dataset."

    # ──────────────────────────────────────────────────────────────────────────
    # Unified public API — single entry point for all query combinations
    # ──────────────────────────────────────────────────────────────────────────

    def recommend(self,
                  title:       str = None,
                  description: str = None,
                  genre:       str = None,
                  top_n:       int = 5) -> tuple[list[dict], str | None]:
        """
        Recommend movies using any combination of title, description, and genre.

        Parameters
        ----------
        title        : Movie title to find similar movies for.
        description  : Free-text plot / mood query.
        genre        : Genre name — hard filter applied AFTER scoring.
        top_n        : How many results to return (default 5).

        At least one parameter must be provided.

        Combination logic
        -----------------
        title only            uses cached title similarity row
        description only      vectorises query, single dot product
        title + description   averages both score vectors (equal weight)
        genre only            top-rated movies in genre, no cosine
        [title|desc] + genre  cosine scoring then genre hard-filter
        all three             averaged cosine scores then genre filter
        """
        if not any([title, description, genre]):
            return [], "Provide at least one of: title, description, genre."

        # genre-only path: skip cosine entirely
        if genre and not title and not description:
            return self._genre_only(genre, top_n)

        # build one score vector per signal, then average
        score_vecs  = []
        exclude_idx = None

        if title:
            idx, err = self._resolve_title(title)
            if err:
                return [], err
            exclude_idx = idx                          # don't recommend the query movie itself
            score_vecs.append(self._get_sim_row(idx))  # cached dot product

        if description:
            score_vecs.append(self._desc_scores(description))  # live dot product

        # equal-weight average of all signals
        scores  = np.mean(score_vecs, axis=0)
        pairs   = self._apply_filters(scores, exclude_idx=exclude_idx, genre=genre)
        results = self._format(pairs, top_n)

        if not results:
            avail = sorted(self.df["genre"].unique())
            return [], (f"No results after filtering by genre='{genre}'. "
                        f"Available genres: {avail}")
        return results, None

    def _genre_only(self, genre: str, top_n: int) -> tuple[list[dict], str | None]:
        subset = self.df[self.df["genre"].str.lower() == genre.lower()]
        if subset.empty:
            avail = sorted(self.df["genre"].unique())
            return [], f"Genre '{genre}' not found. Available: {avail}"
        top = subset.nlargest(top_n, "rating")
        return [
            {"title":      r["title"],
             "year":       r.get("year", ""),
             "genre":      r["genre"],
             "rating":     r["rating"] if pd.notna(r["rating"]) else None,
             "similarity": None,
             "description": r["description"][:150]}
            for _, r in top.iterrows()
        ], None

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    def genres(self) -> list[str]:
        return sorted(self.df["genre"].unique())

    def stats(self) -> dict:
        return {
            "total_movies": len(self.df),
            "genres":       self.df["genre"].nunique(),
            "vocab_size":   len(self.vectorizer.vocabulary_),
            "matrix_shape": self.tfidf_matrix_norm.shape,
            "cache_size":   len(self._sim_cache),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Demo — every query combination
# ──────────────────────────────────────────────────────────────────────────────

def _print(results, err, header):
    print(f"\n  {'─'*62}")
    print(f"  {header}")
    print(f"  {'─'*62}")
    if err:
        print(f"  ERROR: {err}")
        return
    for i, r in enumerate(results, 1):
        rating = f"  rating={r['rating']}" if r["rating"] else ""
        sim    = f"  sim={r['similarity']:.4f}" if r["similarity"] is not None else ""
        print(f"  {i}. {r['title']} ({r['year']})  [{r['genre']}]{rating}{sim}")
        print(f"     {r['description'][:110]}...")


if __name__ == "__main__":
    rec = MovieRecommender()
    s = rec.stats()
    print(f"\n{'='*62}")
    print(f"  {s['total_movies']:,} movies | {s['genres']} genres | "
          f"vocab {s['vocab_size']:,} | matrix {s['matrix_shape']}")
    print(f"{'='*62}")

    # 1 param: title only
    r, e = rec.recommend(title="13 Cameras")
    _print(r, e, "[1 param]  title='13 Cameras'")

    # 1 param: description only
    r, e = rec.recommend(description="a detective hunting a serial killer")
    _print(r, e, "[1 param]  description='a detective hunting a serial killer'")

    # 1 param: genre only
    r, e = rec.recommend(genre="Horror")
    _print(r, e, "[1 param]  genre='Horror'")

    # 2 params: title + genre
    r, e = rec.recommend(title="13 Cameras", genre="Thriller")
    _print(r, e, "[2 params] title='13 Cameras' + genre='Thriller'")

    # 2 params: description + genre
    r, e = rec.recommend(description="heist bank robbery crew", genre="Action")
    _print(r, e, "[2 params] description='heist bank robbery crew' + genre='Action'")

    # 2 params: title + description
    r, e = rec.recommend(title="13 Cameras",
                         description="couple stalked in their own home")
    _print(r, e, "[2 params] title='13 Cameras' + description='couple stalked in their home'")

    # 3 params: all three
    r, e = rec.recommend(title="13 Cameras",
                         description="surveillance landlord spying newlywed",
                         genre="Thriller")
    _print(r, e, "[3 params] title + description + genre='Thriller'  (ALL THREE)")

    # cache benchmark
    t0 = time.time()
    rec.recommend(title="13 Cameras")
    print(f"\n  2nd call same title (cache hit): {(time.time()-t0)*1000:.2f}ms")