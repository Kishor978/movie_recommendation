"""
movie_cli.py – Interactive command-line interface for the Movie Recommender
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from recommender import build_dataframe, RAW_MOVIES, MovieRecommender, evaluate


def print_results(results, header="Top Recommendations"):
    print(f"\n{'─'*60}")
    print(f"  {header}")
    print(f"{'─'*60}")
    for i, r in enumerate(results, 1):
        sim_str = f"  similarity={r['similarity']:.4f}" if r.get("similarity") is not None else ""
        print(f"  {i}. {r['title']}")
        print(f"     Genre: {r['genre']}  |  Rating: ⭐{r['rating']}{sim_str}")
        if "description_snippet" in r:
            print(f"     ↳ {r['description_snippet']}")
    print()


def main():
    df = build_dataframe(RAW_MOVIES)
    rec = MovieRecommender(df)
    genres = sorted(df["genre"].unique())

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║       🎬  Movie Recommendation System (TF-IDF)       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  {len(df)} movies loaded across {len(genres)} genres")
    print(f"  Available genres: {', '.join(genres)}\n")

    while True:
        print("Choose search mode:")
        print("  [1] By movie title")
        print("  [2] By vague description")
        print("  [3] By genre (top-rated)")
        print("  [4] Run evaluation metrics")
        print("  [5] Exit")
        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            title = input("  Enter movie title: ").strip()
            gf = input("  Filter by genre? (press Enter to skip): ").strip() or None
            results, err = rec.by_title(title, genre_filter=gf)
            if err:
                print(f"  ❌ {err}")
            else:
                print_results(results, f"Movies similar to '{title}'")

        elif choice == "2":
            query = input("  Describe what you want to watch: ").strip()
            gf = input("  Filter by genre? (press Enter to skip): ").strip() or None
            results, _ = rec.by_description(query, genre_filter=gf)
            print_results(results, f"Matches for: '{query}'")

        elif choice == "3":
            genre = input(f"  Enter genre ({'/'.join(genres)}): ").strip()
            results, err = rec.by_genre(genre)
            if err:
                print(f"  ❌ {err}")
            else:
                print_results(results, f"Top {genre} Movies")

        elif choice == "4":
            print("\n  Computing evaluation metrics…")
            p_at_5 = evaluate(rec, k=5)
            p_at_3 = evaluate(rec, k=3)
            p_at_10 = evaluate(rec, k=10)
            print(f"\n  Precision@3   : {p_at_3:.4f} ({p_at_3*100:.1f}%)")
            print(f"  Precision@5   : {p_at_5:.4f} ({p_at_5*100:.1f}%)")
            print(f"  Precision@10  : {p_at_10:.4f} ({p_at_10*100:.1f}%)")
            print("  (Genre agreement used as proxy for relevance)\n")

        elif choice == "5":
            print("\n  Goodbye! 🎬\n")
            break

        else:
            print("  Invalid choice, try again.\n")


if __name__ == "__main__":
    main()
