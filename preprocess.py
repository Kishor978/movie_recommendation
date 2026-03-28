"""
preprocess.py — shared text cleaning used by both build.py and recommender.py

Keeping this in one place ensures the EXACT same transformation is applied
at build time (corpus) and at query time (user input). If they ever diverge,
cosine scores become meaningless.
"""

import re

# ── Stopwords (no NLTK dependency) ──────────────────────────────────────────
STOPWORDS = set("""
a about above after again against all am an and any are arent as at be
because been before being below between both but by cant cannot could
couldnt did didnt do does doesnt doing dont down during each few for
from further get got had hadnt has hasnt have havent having he hed
hell hes her here heres hers herself him himself his how hows i id
ill im ive if in into is isnt it its itself lets me more most
mustnt my myself no nor not of off on once only or other ought our ours
ourselves out over own same shant she shed shell shes should shouldnt
so some such than that thats the their theirs them themselves then there
theres these they theyd theyll theyre theyve this those through to too
under until up very was wasnt we wed well were weve were werent what
whats when whens where wheres which while who whos whom why whys will
with wont would wouldnt you youd youll youre youve your yours yourself
yourselves also one two three film movie story man woman young old new just
see full summary
""".split())


def clean_text(text: str) -> str:
    """
    Lowercase → strip punctuation → remove stopwords → drop short tokens.
    Applied identically at build time and query time.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation / numbers
    text = re.sub(r"\s+", " ", text).strip()     # collapse whitespace
    tokens = [t for t in text.split()
              if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def parse_title(raw: str) -> str:
    """
    'Inception - 2010'  ->  'Inception'
    'Some Movie - nan'  ->  'Some Movie'
    'None - 2015'       ->  None  (dropped downstream)
    """
    if not isinstance(raw, str):
        return None
    parts = raw.rsplit(" - ", 1)
    title = parts[0].strip()
    if title.lower() in ("none", "nan", ""):
        return None
    return title


def parse_year(raw: str) -> str:
    """
    'Inception - 2010'  →  '2010'
    'Some Movie - nan'  →  ''
    """
    if not isinstance(raw, str):
        return ""
    parts = raw.rsplit(" - ", 1)
    if len(parts) == 2 and parts[1].strip().lower() != "nan":
        return parts[1].strip()
    return ""


def load_and_clean_dataset(csv_path: str):
    """
    Load the raw CSV, parse columns, drop rows with empty descriptions,
    and return a clean DataFrame ready for vectorisation.

    Columns in output:
        title, year, genre, expanded_genres, rating, description, clean_desc
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # ── Rename columns to safe names ────────────────────────────────────────
    df = df.rename(columns={
        "movie title - year": "raw_title",
        "expanded-genres":    "expanded_genres",
    })

    # ── Parse title and year out of combined column ──────────────────────────
    df["title"] = df["raw_title"].apply(parse_title)
    df["year"]  = df["raw_title"].apply(parse_year)
    df = df.drop(columns=["raw_title"])

    # ── Strip "See full summary »" artefact from truncated descriptions ──────
    df["description"] = (
        df["description"]
        .fillna("")
        .str.replace(r"\s*\.{3}\s*See full summary\s*»?", "", regex=True)
        .str.strip()
    )

    # ── Drop rows where description is empty after cleaning ─────────────────
    df = df[df["description"].str.len() > 10].copy()

    # ── Enrich text: prepend expanded genre tags before description ───────────
    # e.g.  'crime drama horror  A newlywed couple move into a house ...'
    # Genre words like 'thriller crime horror' massively increase meaningful
    # TF-IDF overlap between thematically similar movies.
    # We repeat them (x3) so they carry enough IDF weight vs the description.
    def enrich(row):
        tags = " ".join(
            g.strip().lower()
            for g in str(row["expanded_genres"]).split(",")
            if isinstance(row["expanded_genres"], str)
        )
        tags_repeated = (tags + " ") * 3   # repeat so genre terms get weight
        return tags_repeated + row["description"]

    df["enriched_text"] = df.apply(enrich, axis=1)

    # ── Clean the enriched text (vectoriser input) ───────────────────────────
    df["clean_desc"] = df["enriched_text"].apply(clean_text)

    # ── Drop rows where clean_desc is empty (extremely short / all stopwords) 
    df = df[df["clean_desc"].str.len() > 0].copy()

    # ── Drop rows with null title (edge case: 1 row has no parseable title) ──
    df = df[df["title"].notna() & (df["title"].str.strip() != "")].copy()

    # ── Reset index so row positions match matrix rows exactly ──────────────
    df = df.reset_index(drop=True)

    return df