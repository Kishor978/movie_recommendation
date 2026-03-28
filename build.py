"""
build.py — MODE 1: Run ONCE to build and persist all model artifacts.

What this script does
─────────────────────
1. Load + clean the raw CSV dataset
2. Fit TF-IDF vectoriser on all movie descriptions
3. L2-normalise the resulting sparse matrix
4. Save three artifacts to model/:
       vectorizer.joblib        — fitted TfidfVectorizer
       tfidf_matrix.npz         — L2-normalised sparse matrix  (scipy .npz)
       movies.parquet           — cleaned DataFrame (title, genre, rating …)

After this script finishes, build.py never needs to run again unless the
dataset changes. recommender.py loads the saved files in milliseconds.

Usage
─────
    python3 build.py --input data/test.csv --model-dir model/
    python3 build.py                          # uses defaults below
"""

import argparse
import os
import time

import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from preprocess import load_and_clean_dataset


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_INPUT     = "/mnt/user-data/uploads/test.csv"
DEFAULT_MODEL_DIR = "model"

# ── TF-IDF hyper-parameters ─────────────────────────────────────────────────
TFIDF_CONFIG = dict(
    max_features = 15000,     # larger vocab for 30k real movies vs 80 dummy ones
    ngram_range  = (1, 2),    # unigrams + bigrams
    sublinear_tf = True,      # log(tf) dampens very common terms
    min_df       = 2,         # ignore terms that appear in only 1 doc (noise)
    max_df       = 0.95,      # ignore terms in >95% of docs (near-stopwords)
)


def build(input_path: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    # ── 1. Load & clean ──────────────────────────────────────────────────────
    print(f"[1/4] Loading dataset from: {input_path}")
    t0 = time.time()
    df = load_and_clean_dataset(input_path)
    print(f"      {len(df):,} movies loaded in {time.time()-t0:.1f}s")
    print(f"      Genres : {df['genre'].nunique()} unique")
    print(f"      Ratings: {df['rating'].notna().sum():,} present, "
          f"{df['rating'].isna().sum():,} missing")

    # ── 2. Fit TF-IDF ────────────────────────────────────────────────────────
    print(f"\n[2/4] Fitting TF-IDF vectoriser ...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    tfidf_raw = vectorizer.fit_transform(df["clean_desc"])
    sparsity = 1 - tfidf_raw.nnz / (tfidf_raw.shape[0] * tfidf_raw.shape[1])
    print(f"      Vocabulary size : {len(vectorizer.vocabulary_):,} terms")
    print(f"      Matrix shape    : {tfidf_raw.shape}  (movies x features)")
    print(f"      Sparsity        : {sparsity:.1%}")
    print(f"      Done in {time.time()-t0:.1f}s")

    # ── 3. L2-normalise rows once ────────────────────────────────────────────
    #
    #   After normalisation: cosine_sim(row_i, row_j) = row_i . row_j
    #   This means ALL future similarity queries are just a single dot product.
    #   No norm computation needed at query time ever again.
    #
    print(f"\n[3/4] L2-normalising matrix ...")
    t0 = time.time()
    tfidf_norm = normalize(tfidf_raw, norm="l2")   # still sparse CSR
    print(f"      Done in {time.time()-t0:.2f}s")

    # ── 4. Persist artifacts ─────────────────────────────────────────────────
    print(f"\n[4/4] Saving artifacts to '{model_dir}/' ...")
    t0 = time.time()

    # 4a. Vectoriser — joblib handles sklearn objects natively
    vec_path = os.path.join(model_dir, "vectorizer.joblib")
    joblib.dump(vectorizer, vec_path, compress=3)
    print(f"      vectorizer.joblib    -> {os.path.getsize(vec_path)/1024:.0f} KB")

    # 4b. Sparse matrix — scipy .npz is the native format for sparse arrays.
    #     Far more compact than joblib for large sparse matrices.
    #     Stores the CSR arrays (data, indices, indptr) compressed.
    mat_path = os.path.join(model_dir, "tfidf_matrix.npz")
    sp.save_npz(mat_path, tfidf_norm)
    print(f"      tfidf_matrix.npz     -> {os.path.getsize(mat_path)/1024/1024:.1f} MB")

    # 4c. DataFrame — save as compressed CSV (parquet needs pyarrow/fastparquet).
    #     In production swap to .to_parquet() for ~10x faster loads.
    #     index=True is critical: the row index == the matrix row index.
    df_save = df[["title", "year", "genre", "expanded_genres",
                  "rating", "description"]].copy()
    df_path = os.path.join(model_dir, "movies.csv.gz")
    df_save.to_csv(df_path, index=True, compression="gzip")
    print(f"      movies.csv.gz        -> {os.path.getsize(df_path)/1024/1024:.1f} MB")

    print(f"\n      All artifacts saved in {time.time()-t0:.2f}s")
    print(f"\n  Build complete. Run recommender.py to serve queries.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build movie recommender artifacts")
    parser.add_argument("--input",     default=DEFAULT_INPUT,     help="Path to raw CSV")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Where to save artifacts")
    args = parser.parse_args()
    build(args.input, args.model_dir)