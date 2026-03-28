# movie_recommendation

# Movie Recommendation System
### AI/ML Intern Task — Content-Based Filtering with TF-IDF + Cosine Similarity

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Setup & Usage](#4-setup--usage)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Vectorisation — TF-IDF](#6-vectorisation--tf-idf)
7. [Recommendation Logic](#7-recommendation-logic)
8. [Artifact Persistence — Build Once, Serve Forever](#8-artifact-persistence--build-once-serve-forever)
9. [Query API](#9-query-api)
10. [Evaluation Results](#10-evaluation-results)
11. [Limitations](#11-limitations)
12. [How to Improve](#12-how-to-improve)

---

## 1. Project Overview

A **content-based movie recommendation system** built with Python, Pandas, and Scikit-learn.

Movies are recommended based on the **semantic similarity of their plot descriptions and genre tags**, vectorised using TF-IDF and ranked using cosine similarity. No pre-trained embeddings are used.

The system separates **building** (offline, run once) from **serving** (online, instant). Once the TF-IDF matrix is built and saved to disk, every subsequent query is a single sparse dot product against the stored matrix — no recomputation, no re-fitting, no model rebuild.

---

## 2. Dataset

**Source:** `jquigl/imdb-genres` on HuggingFace  
**Local file:** `train.csv` (238,237 movies after cleaning)

| Column | Type | Usage |
|--------|------|-------|
| `movie title - year` | string | Parsed into `title` + `year` |
| `genre` | string | Primary genre label — used for genre-only queries and evaluation |
| `expanded-genres` | string | Multi-label genre list (e.g. `"Crime, Drama, Thriller"`) — prepended to description before vectorisation; used as ground truth in evaluation |
| `rating` | float | IMDB rating — used for genre-only ranking |
| `description` | string | Plot summary — primary content signal |

**Dataset facts (after cleaning):**
- 238,237 movies across 16 genres
- 70% of movies have ratings present; 30% are unrated
- Average description length: ~169 characters (~30 words post-cleaning)
- 83% of movies have multi-genre labels (avg 2.37 genres per movie)

---

## 3. Project Structure

```
movie-recommendation-system/
│
├── preprocess.py        # Shared text cleaning (used by build AND recommender)
├── build.py             # MODE 1: fit TF-IDF, save artifacts — run once
├── recommender.py       # MODE 2: load artifacts, serve all queries
├── evaluation.py        # Metrics: Precision@K, Recall@K, Coverage, Diversity, Latency
│
└── model/               # Saved artifacts (output of build.py)
    ├── vectorizer.joblib      # Fitted TfidfVectorizer (~169 KB)
    ├── tfidf_matrix.npz       # L2-normalised sparse matrix (~40 MB at 238k movies)
    └── movies.csv.gz          # DataFrame: titles, genres, ratings, descriptions
```

---

## 4. Setup & Usage

**Requirements**

```
python >= 3.10
pandas
scikit-learn
scipy
joblib
numpy
```

**Step 1 — Build (run once)**

Fits the TF-IDF vectoriser on the full corpus and saves all artifacts to `model/`.

```bash
python build.py --input test.csv --model-dir model/
```

**Step 2 — Query (every time)**

Loads from disk in ~3 seconds. Every query is then milliseconds.

```bash
python recommender.py        # runs the built-in demo
python evaluation.py --sample 500   # runs all metrics on a 500-movie sample
```

---

## 5. Data Preprocessing

All preprocessing lives in `preprocess.py` and is applied **identically** at build time (to the corpus) and at query time (to user input). If these ever diverge, cosine scores become meaningless.

### Pipeline

```
raw description
    │
    ├─ Strip "See full summary »" truncation artefact
    │
    ├─ Prepend expanded genre tags × 3
    │     "crime drama horror crime drama horror crime drama horror
    │      A newlywed couple move into a new house..."
    │
    ├─ Lowercase
    │
    ├─ Remove punctuation and numbers  [^a-z\s] → space
    │
    ├─ Remove stopwords  (~180 common English words, hardcoded — no NLTK needed)
    │
    └─ Drop tokens with length ≤ 2
```

### Why prepend genre tags?

This is the single most impactful design decision in the system.

TF-IDF relies on **word overlap**. Plot descriptions average ~30 words after cleaning, and two thematically identical movies described with different words score near-zero similarity. Genre labels from `expanded-genres` (e.g. `"crime drama thriller"`) are the strongest shared vocabulary between movies with similar themes. Prepending them 3× gives them enough weight to anchor the similarity scores.

**Effect on scores:**

| | Without genre enrichment | With genre enrichment |
|--|--|--|
| Mean cosine score (top-10) | 0.24 | 0.52 |
| Precision@5 | 16.2% | 91.9% |
| Zero-score results | >50% | 0% |

Repeating the tags 3× prevents TF-IDF's IDF component from penalising them too heavily just because they appear in many documents (which genre words naturally do).

### Title parsing

The raw column `"Inception - 2010"` is split on ` - ` into `title="Inception"` and `year="2010"`. Literal `"None"` and `"nan"` titles are dropped (1 row).

### Stopwords

A 180-word custom set, hardcoded — no NLTK or external library required. Includes domain-specific tokens like `"film"`, `"movie"`, `"story"`, `"see"`, `"full"`, `"summary"` that are uninformative in plot descriptions.

### Why no stemming or lemmatisation?

TF-IDF with bigrams (`ngram_range=(1,2)`) already captures morphological variants through co-occurrence. Adding stemming (e.g. Porter) would require NLTK and can introduce noise on film-specific vocabulary. The genre enrichment strategy gives better signal per token than stemming would.

---

## 6. Vectorisation — TF-IDF

### Why TF-IDF over Bag-of-Words?

| Property | TF-IDF | BoW (raw counts) |
|----------|--------|-----------------|
| Common-word dominance | Suppressed via IDF | Unchecked |
| Rare / distinctive terms | Up-weighted | Equal footing |
| Length sensitivity | Log-normalised (`sublinear_tf`) | Long docs score higher |
| Interpretability | High | High |

TF-IDF rewards words that are *distinctive* to a movie (high TF) but *rare* across the corpus (high IDF). This is exactly what's needed for plot summaries: words like `"heist"`, `"haunted"`, `"serial"` are more informative than words like `"man"` or `"life"`.

### Configuration

```python
TfidfVectorizer(
    max_features = 15000,   # vocabulary cap — covers 99%+ of useful terms
    ngram_range  = (1, 2),  # unigrams + bigrams: captures "serial killer", "time travel"
    sublinear_tf = True,    # tf = 1 + log(tf), dampens very frequent terms
    min_df       = 2,       # drop terms appearing in only 1 document (noise/typos)
    max_df       = 0.95,    # drop terms in >95% of documents (near-stopwords)
)
```

### L2 Normalisation

After fitting, every row is L2-normalised:

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
                        = (a · b)   ← when both are unit vectors
```

This means **every query is a dot product** — no norm computation at query time, ever. The normalised matrix is saved to disk and used as-is for all queries.

---

## 7. Recommendation Logic

### Architecture

```
BUILD (once)                         SERVE (every query)
──────────────────────────────       ──────────────────────────────────────
raw CSV                              user input: title / description / genre
  │                                    │
  ├─ preprocess.py                     ├─ clean_text()  [same as build]
  ├─ TfidfVectorizer.fit_transform     ├─ vectorizer.transform()  [no fit]
  ├─ normalize(matrix, norm="l2")      ├─ normalize(query_vec)
  ├─ save vectorizer.joblib            ├─ query_vec @ tfidf_matrix_norm.T
  ├─ save tfidf_matrix.npz             ├─ (optional) genre index filter
  └─ save movies.csv.gz                └─ top-N results
```

### Cosine similarity as dot product

Because both the corpus matrix and query vectors are L2-normalised:

```
similarity(query, movie_i) = query_vec · matrix_row_i
```

This is a single sparse matrix multiply — the most efficient possible operation for this architecture.

### Lazy similarity cache (title queries)

When a title query is made, the system computes one row of the similarity matrix and stores it:

```python
scores = (matrix_row[idx] @ tfidf_matrix_norm.T).flatten()
_sim_cache[idx] = scores   # shape (N,)
```

Every subsequent query for the same title is an O(1) dict lookup. The cache grows as titles are queried — never pre-computed in full (that would be N² memory).

### Genre index

A dict mapping `genre_lower → set of row indices` is built at startup. Genre filtering is an O(1) set intersection rather than a DataFrame `.loc` scan, which reduced the "all three params" query time from 413ms to ~15ms on the 30k dataset (410ms on 238k).

---

## 8. Artifact Persistence — Build Once, Serve Forever

### Why persist?

On 238,237 movies, fitting and normalising TF-IDF takes ~30 seconds. This is unacceptable on every server restart or cold start. The three artifacts encode all necessary state:

| Artifact | Format | Size (238k movies) | What it contains |
|----------|--------|---------------------|------------------|
| `vectorizer.joblib` | joblib compressed | ~169 KB | Vocabulary, IDF weights, sklearn object |
| `tfidf_matrix.npz` | scipy sparse NPZ | ~40 MB | L2-normalised (238k × 15k) CSR matrix |
| `movies.csv.gz` | gzip CSV | ~20 MB | Titles, genres, ratings, descriptions |

**Total on disk: ~60 MB. Load time: ~3 seconds.**

### Format choices

- **joblib** for the vectoriser: native sklearn serialisation, handles all internal state correctly.
- **scipy .npz** for the matrix: the native format for sparse arrays. Stores CSR components (`data`, `indices`, `indptr`) in compressed numpy format. Far smaller than joblib for large sparse matrices.
- **csv.gz** for the DataFrame: no external dependency. Swap to **parquet** (`pip install pyarrow`) for ~5× faster DataFrame load in production.

### Two-mode workflow

```
MODE 1 — BUILD (run once, or when dataset changes)
  python build.py --input test.csv --model-dir model/

MODE 2 — SERVE (every deployment restart / every query)
  rec = MovieRecommender(model_dir="model/")   # loads in ~3s
  results, err = rec.recommend(title="Inception")  # ~500ms first call, ~300ms cached
```

---

## 9. Query API

All queries go through a single method:

```python
results, error = rec.recommend(
    title       = None,   # movie title string
    description = None,   # free-text plot / mood description
    genre       = None,   # genre name (hard filter)
    top_n       = 5       # number of results
)
```

**At least one parameter must be provided.** All 7 combinations are supported:

```python
# 1 parameter
rec.recommend(title="Inception")
rec.recommend(description="a detective hunting a serial killer")
rec.recommend(genre="Horror")

# 2 parameters
rec.recommend(title="Inception", genre="Scifi")
rec.recommend(description="heist bank robbery crew", genre="Action")
rec.recommend(title="Inception", description="mind-bending dream heist")

# 3 parameters
rec.recommend(title="Inception", description="dream heist layers", genre="Scifi")
```

### Signal combination logic

| Parameters | Strategy |
|-----------|----------|
| `title` only | Cached sim row — O(1) after first call |
| `description` only | Transform query once, single dot product |
| `title + description` | Average both score vectors (equal weight) |
| `genre` only | Top-rated movies in genre — no cosine |
| `[title\|desc] + genre` | Cosine scores → genre hard-filter via index |
| All three | Average cosine scores → genre hard-filter |

### Result format

Each result is a dict:

```python
{
    "title":       "Se7en",
    "year":        "1995",
    "genre":       "Thriller",
    "rating":      8.6,
    "similarity":  0.5301,
    "description": "Two detectives hunt a serial killer..."
}
```

---

## 10. Evaluation Results

Evaluated on a random sample of **500 movies** from the full 238,237-movie corpus (seed=42).

### Ground truth

**Multi-label** genre matching via `expanded-genres`. A recommendation is a **hit** if the recommended movie shares **any** genre with the query movie. This reflects real relevance: a `"Crime, Drama, Thriller"` movie recommended for a `"Thriller"` query is a genuine match, not a miss.

### Metric 1 — Precision@K

*Of the top-K recommendations, what fraction share at least one genre with the query movie?*

| K | Precision | Interpretation |
|---|-----------|----------------|
| 1 | **93.4%** | The top result is genre-relevant 93% of the time |
| 3 | **92.1%** | 2.76 out of 3 results are genre-relevant on average |
| 5 | **91.9%** | 4.6 out of 5 results are genre-relevant on average |
| 10 | **91.9%** | Stable — the model doesn't degrade at larger K |

### Metric 2 — Recall@K

*Of all genre-relevant movies in the corpus, what fraction appear in the top-K results?*

| K | Recall |
|---|--------|
| 5 | **91.9%** |
| 10 | **91.9%** |

Recall is bounded by the extremely large genre pools (e.g. 36,215 Thrillers). Top-5 can only ever surface a tiny fraction of all relevant movies — but those 5 are almost always relevant.

### Metric 3 — Coverage@5

**0.5%** of the catalogue appears in at least one recommendation list (evaluated on 500 queries).

This is expected and not a concern: with 238k movies and only 500 query movies × 5 recs = 2,500 slots, full catalogue coverage is impossible. On a production system with millions of queries, coverage naturally increases. The low figure reflects the sample size, not a popularity bias.

### Metric 4 — Intra-list Genre Diversity@5

**2.10** unique genres per recommendation list on average (max possible = 5).

This reflects the genre enrichment strategy: movies with similar genre combinations cluster tightly together, so recommendations naturally stay within 2–3 genre families. This is desirable for precision but reduces cross-genre serendipity.

### Metric 5 — Self-similarity Sanity Check

**500/500 PASS** — every movie's cosine similarity with itself equals 1.0, confirming L2 normalisation is intact after loading from disk.

### Metric 6 — Score Distribution (top-10 recs, 100 queries)

| Stat | Value |
|------|-------|
| Mean cosine score | 0.5163 |
| Median | 0.4784 |
| Min | 0.2366 |
| Max | 1.0 |
| Zero scores | 0.0% |

Scores are well-distributed with no zero-score dead zones — a direct result of the genre enrichment strategy which guarantees token overlap between genre-matched movies.

### Metric 7 — Per-genre Precision@5

| Genre | Precision | n |
|-------|-----------|---|
| Animation | **100.0%** | 11 |
| History | **100.0%** | 15 |
| Romance | 96.8% | 62 |
| Adventure | 95.6% | 32 |
| Crime | 95.1% | 57 |
| Fantasy | 95.0% | 20 |
| Mystery | 92.4% | 29 |
| War | 91.4% | 14 |
| Action | 90.8% | 65 |
| Thriller | 89.1% | 77 |
| Family | 88.2% | 17 |
| Scifi | 88.0% | 25 |
| Biography | 87.7% | 13 |
| Horror | 87.4% | 54 |
| Sports | 77.8% | 9 |

**Sports** scores lowest (77.8%) because sports movies span wildly different sub-themes (boxing dramas, football comedies, racing action) with minimal shared plot vocabulary. **Animation** and **History** score 100% because their descriptions share strong distinctive vocabulary.

### Metric 8 — Query Latency (238k movies)

| Query mode | Latency |
|-----------|---------|
| Title — cache miss (1st call) | ~527 ms |
| Title — cache hit (2nd call) | ~315 ms |
| Description — live dot product | ~383 ms |
| Genre only — no cosine | ~130 ms |
| Title + Description — averaged | ~515 ms |
| Title + Description + Genre | ~410 ms |

Latency scales with corpus size. At 238k movies the dot product (`1 × 238,237 × 15,000` sparse) takes ~400ms. This is acceptable for a batch or API setting. For sub-100ms latency at this scale, approximate nearest-neighbour search (FAISS) would be required.

---

## 11. Limitations

### 1. Latency scales with corpus size
The dot product against 238k rows takes ~400ms. TF-IDF cosine search is exact but linear in N. This becomes a bottleneck beyond ~500k movies.

### 2. No semantic understanding
TF-IDF treats words as independent tokens. `"detective hunting killer"` and `"investigator pursuing murderer"` have near-zero similarity despite identical meaning. Pre-trained sentence embeddings (Sentence-BERT) would solve this but are excluded per task requirements.

### 3. Description quality dependency
Recommendations are entirely dependent on description richness. Short (< 15 word) descriptions, non-English text, and stub summaries produce weak vectors. 30% of movies have no rating, which limits genre-only ranking quality.

### 4. Genre enrichment creates genre clustering
Prepending genre tags improves precision dramatically but reduces cross-genre serendipity. A Horror-Comedy will reliably recommend other Horror-Comedies rather than surfacing thematically related dramas.

### 5. Static matrix
The TF-IDF matrix is built once at startup. Adding new movies requires a full rebuild (`build.py`) which takes ~30 seconds on 238k movies.

### 6. Cold start — new movies
A new movie with no description cannot be vectorised. It also won't appear in recommendations until the next build.

### 7. Evaluation metric is a proxy
Genre-match precision is a proxy for relevance, not a direct measure of user satisfaction. A movie with cosine similarity 0.95 that happens to differ in primary genre would count as a miss even if it's objectively a great recommendation.

---

## 12. How to Improve

| Area | Current | Improvement |
|------|---------|-------------|
| **Semantic understanding** | TF-IDF word overlap | Sentence-BERT or OpenAI embeddings |
| **Latency at scale** | Exact linear scan ~400ms | FAISS ANN index — sub-10ms at any scale |
| **Missing rating fill** | Unrated movies skipped in genre-only | Predict rating from description using regression |
| **Hybrid signals** | Content only | Add collaborative filtering (user watch history) |
| **Rating-weighted ranking** | Cosine score only | `final_score = α × cosine + (1-α) × normalized_rating` |
| **Incremental updates** | Full rebuild on new data | HashingVectorizer for online updates |
| **DataFrame storage** | csv.gz | parquet (pyarrow) — 5× faster load |
| **Stemming** | None | spaCy lemmatization to collapse `kill/killing/killer` |
| **User feedback loop** | None | Re-rank using click-through data (learning-to-rank) |
| **Multi-language** | English only | multilingual-e5 embeddings |