"""
evaluation.py — Metrics for the Movie Recommendation System

Metrics
───────
1. Precision@K      Genre-match rate in top-K results.
                    Proxy for relevance: same-genre results are counted as hits.
                    Computed for K = 1, 3, 5, 10.

2. Genre Recall@K   Of all movies in the query's genre, what fraction
                    appear in the top-K recommendations on average.

3. Coverage@K       What % of the catalogue ever appears in any rec list.
                    Low coverage = popularity bias (same movies recommended always).

4. Intra-list       Avg number of unique genres per recommendation list.
Diversity           Higher = more diverse / serendipitous results.

5. Self-similarity  Sanity check: every movie should have cosine_sim = 1.0
                    with itself (because rows are L2-normalised).

6. Query timing     Latency for title (cache miss), title (cache hit),
                    description, and combined queries.

Usage
─────
    python evaluation.py                  # evaluate on all movies (slow ~30s)
    python evaluation.py --sample 500     # evaluate on random 500-movie sample
    python evaluation.py --sample 500 --seed 99
"""

import argparse
import time
import random
import numpy as np
import pandas as pd

from recommender import MovieRecommender


# ──────────────────────────────────────────────────────────────────────────────
# Metric functions
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(rec: MovieRecommender, k: int, sample_idx: list[int]) -> float:
    """
    For each sampled movie, get top-K recommendations and count genre matches.
    Precision@K = (# genre matches) / K, averaged over all sampled movies.
    """
    scores = []
    for idx in sample_idx:
        row   = rec.df.loc[idx]
        genre = row["genre"]
        title = row["title"]
        results, err = rec.recommend(title=title, top_n=k)
        if err or not results:
            continue
        hits = sum(1 for r in results if r["genre"] == genre)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(rec: MovieRecommender, k: int, sample_idx: list[int]) -> float:
    """
    Recall@K = (# same-genre movies in top-K) / (total movies in that genre - 1)
    Averaged over all sampled movies.
    """
    genre_counts = rec.df["genre"].value_counts().to_dict()
    scores = []
    for idx in sample_idx:
        row        = rec.df.loc[idx]
        genre      = row["genre"]
        title      = row["title"]
        total_same = genre_counts.get(genre, 1) - 1   # exclude the movie itself
        if total_same == 0:
            continue
        results, err = rec.recommend(title=title, top_n=k)
        if err or not results:
            continue
        hits = sum(1 for r in results if r["genre"] == genre)
        scores.append(hits / min(k, total_same))
    return float(np.mean(scores)) if scores else 0.0


def coverage_at_k(rec: MovieRecommender, k: int, sample_idx: list[int]) -> float:
    """
    Fraction of the full catalogue that appears in at least one rec list.
    """
    recommended = set()
    for idx in sample_idx:
        title = rec.df.loc[idx, "title"]
        results, err = rec.recommend(title=title, top_n=k)
        if results:
            for r in results:
                recommended.add(r["title"])
    return len(recommended) / len(rec.df)


def intra_list_diversity(rec: MovieRecommender, k: int, sample_idx: list[int]) -> float:
    """
    Average number of unique genres within a single recommendation list.
    Range: 1 (all same genre) to K (all different genres).
    """
    diversities = []
    for idx in sample_idx:
        title = rec.df.loc[idx, "title"]
        results, err = rec.recommend(title=title, top_n=k)
        if results:
            genres = {r["genre"] for r in results}
            diversities.append(len(genres))
    return float(np.mean(diversities)) if diversities else 0.0


def self_similarity_check(rec: MovieRecommender, sample_idx: list[int]) -> tuple[int, int]:
    """
    dot(normalised_row, normalised_row) must equal 1.0 for all rows.
    Verifies the L2-normalisation is intact after loading from disk.
    """
    passed = 0
    for idx in sample_idx:
        row   = rec.tfidf_matrix_norm[idx]
        score = (row @ row.T)[0, 0]
        if abs(score - 1.0) < 1e-5:
            passed += 1
    return passed, len(sample_idx)


def query_timing(rec: MovieRecommender) -> dict:
    """Measure latency for each query mode."""
    timings = {}
    test_title = rec.df.loc[0, "title"]

    # title — cache miss (first call)
    rec._sim_cache.clear()
    t0 = time.perf_counter()
    rec.recommend(title=test_title, top_n=5)
    timings["title_cache_miss_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # title — cache hit (second call)
    t0 = time.perf_counter()
    rec.recommend(title=test_title, top_n=5)
    timings["title_cache_hit_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # description only
    t0 = time.perf_counter()
    rec.recommend(description="detective hunting serial killer", top_n=5)
    timings["description_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # genre only
    t0 = time.perf_counter()
    rec.recommend(genre="Action", top_n=5)
    timings["genre_only_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # title + description (combined)
    t0 = time.perf_counter()
    rec.recommend(title=test_title, description="surveillance stalker home")
    timings["title_plus_desc_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # all three
    t0 = time.perf_counter()
    rec.recommend(title=test_title, description="surveillance stalker home", genre="Thriller")
    timings["all_three_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return timings


def score_distribution(rec: MovieRecommender, sample_idx: list[int], k: int = 10):
    """
    Stats on cosine similarity scores across rec lists.
    Helps understand if scores are meaningful or uniformly near-zero.
    """
    all_scores = []
    for idx in sample_idx[:100]:   # limit to 100 for speed
        title = rec.df.loc[idx, "title"]
        results, err = rec.recommend(title=title, top_n=k)
        if results:
            all_scores.extend([r["similarity"] for r in results
                                if r["similarity"] is not None])
    arr = np.array(all_scores)
    return {
        "mean":   round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "min":    round(float(arr.min()), 4),
        "max":    round(float(arr.max()), 4),
        "pct_zero": round(float((arr == 0).mean() * 100), 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

def bar(value: float, width: int = 30, scale: float = 1.0) -> str:
    filled = int((value / scale) * width)
    return "█" * filled + "░" * (width - filled)


def run_evaluation(model_dir: str = "model", sample: int = None, seed: int = 42):
    print("\n" + "=" * 65)
    print("   Movie Recommendation System — Evaluation Report")
    print("=" * 65)

    rec = MovieRecommender(model_dir)
    n   = len(rec.df)

    # sample indices
    random.seed(seed)
    all_idx = list(range(n))
    sample_idx = random.sample(all_idx, min(sample, n)) if sample else all_idx
    print(f"\n  Evaluating on {len(sample_idx):,} / {n:,} movies  (seed={seed})\n")

    # ── 1. Precision@K ────────────────────────────────────────────────────────
    print("─" * 65)
    print("  [1] Precision@K  (genre-match proxy)")
    print("─" * 65)
    for k in [1, 3, 5, 10]:
        t0 = time.time()
        p  = precision_at_k(rec, k, sample_idx)
        elapsed = time.time() - t0
        print(f"  P@{k:2d}  {p:.4f}  {bar(p)}  ({p*100:.1f}%)  [{elapsed:.1f}s]")

    # ── 2. Recall@K ───────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [2] Recall@K  (genre recall proxy)")
    print("─" * 65)
    for k in [5, 10]:
        t0 = time.time()
        r  = recall_at_k(rec, k, sample_idx)
        elapsed = time.time() - t0
        print(f"  R@{k:2d}  {r:.4f}  {bar(r)}  ({r*100:.1f}%)  [{elapsed:.1f}s]")

    # ── 3. Coverage ───────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [3] Coverage@5  (catalogue coverage)")
    print("─" * 65)
    t0  = time.time()
    cov = coverage_at_k(rec, 5, sample_idx)
    print(f"  {cov*100:.1f}% of catalogue movies appear in at least one "
          f"rec list  [{time.time()-t0:.1f}s]")

    # ── 4. Intra-list Diversity ───────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [4] Intra-list Genre Diversity@5  (avg unique genres per list)")
    print("─" * 65)
    t0  = time.time()
    div = intra_list_diversity(rec, 5, sample_idx)
    print(f"  {div:.2f} unique genres per list on average "
          f"(max possible = 5)  [{time.time()-t0:.1f}s]")

    # ── 5. Self-similarity ────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [5] Self-similarity Sanity Check  (L2 norm integrity)")
    print("─" * 65)
    passed, total = self_similarity_check(rec, sample_idx[:500])
    status = "PASS" if passed == total else f"FAIL ({total - passed} broken rows)"
    print(f"  {passed}/{total} rows have cosine_sim(row, row) = 1.0  [{status}]")

    # ── 6. Score distribution ─────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [6] Cosine Similarity Score Distribution  (top-10 recs, 100 queries)")
    print("─" * 65)
    dist = score_distribution(rec, sample_idx)
    print(f"  mean={dist['mean']}  median={dist['median']}  "
          f"min={dist['min']}  max={dist['max']}  "
          f"zero_scores={dist['pct_zero']}%")
    if dist["pct_zero"] > 50:
        print("  NOTE: >50% zero scores — descriptions may be too short/generic "
              "for TF-IDF overlap. Embeddings would improve this.")

    # ── 7. Per-genre Precision@5 ──────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [7] Per-genre Precision@5")
    print("─" * 65)
    genre_rows = {}
    for idx in sample_idx:
        g = rec.df.loc[idx, "genre"]
        genre_rows.setdefault(g, []).append(idx)

    genre_results = []
    for genre, idxs in sorted(genre_rows.items()):
        p = precision_at_k(rec, 5, idxs)
        genre_results.append((genre, p, len(idxs)))

    genre_results.sort(key=lambda x: x[1], reverse=True)
    for genre, p, cnt in genre_results:
        b = bar(p, width=20)
        print(f"  {genre:12s}  {p:.4f}  {b}  ({p*100:.1f}%)  n={cnt}")

    # ── 8. Query latency ──────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [8] Query Latency")
    print("─" * 65)
    timings = query_timing(rec)
    labels = {
        "title_cache_miss_ms":  "title        (cache miss, 1st call)",
        "title_cache_hit_ms":   "title        (cache hit,  2nd call)",
        "description_ms":       "description  (live dot product)",
        "genre_only_ms":        "genre only   (no cosine)",
        "title_plus_desc_ms":   "title + desc (2 signals, averaged)",
        "all_three_ms":         "title + desc + genre  (all three)",
    }
    for key, label in labels.items():
        ms = timings[key]
        b  = bar(ms, width=20, scale=max(timings.values()))
        print(f"  {label:42s}  {ms:6.2f} ms  {b}")

    # ── 9. Dataset stats ──────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [9] Dataset & Model Stats")
    print("─" * 65)
    s = rec.stats()
    print(f"  Total movies   : {s['total_movies']:,}")
    print(f"  Genres         : {s['genres']}")
    print(f"  Vocab size     : {s['vocab_size']:,} terms")
    print(f"  Matrix shape   : {s['matrix_shape']}")
    nnz = rec.tfidf_matrix_norm.nnz
    total_cells = s['matrix_shape'][0] * s['matrix_shape'][1]
    print(f"  Sparsity       : {(1 - nnz/total_cells)*100:.1f}%  "
          f"({nnz:,} non-zero entries)")

    print("\n" + "─" * 65)
    print("  Genre distribution")
    print("─" * 65)
    for genre, cnt in rec.df["genre"].value_counts().items():
        b = bar(cnt, width=25, scale=rec.df["genre"].value_counts().max())
        print(f"  {genre:12s}  {cnt:5,}  {b}")

    print("\n" + "=" * 65)
    print("  Evaluation complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--sample",    type=int, default=300,
                        help="Number of movies to sample for evaluation (default 300). "
                             "Use 0 for full dataset.")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    sample = args.sample if args.sample > 0 else None
    run_evaluation(model_dir=args.model_dir, sample=sample, seed=args.seed)