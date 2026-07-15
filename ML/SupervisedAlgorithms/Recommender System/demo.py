"""
Demo: Hybrid Book Recommender
==============================
Runs a few example scenarios and saves visualizations to outputs/.

Run:
    python demo.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recommender import HybridBookRecommender
from evaluate import evaluate

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_recs(title, recs):
    print(f"\n{title}")
    print("-" * len(title))
    for i, r in enumerate(recs, 1):
        print(f"{i:2d}. {r['title']}  —  {r['authors']}  (score={r['score']:.2f})")
        print(f"     why: {r['why']}")


def plot_score_breakdown(rec, uid, filename="01_score_breakdown.png"):
    """Show how alpha (CF vs content weighting) shifts the recommendation mix."""
    alphas = [0.0, 0.3, 0.6, 1.0]
    fig, axes = plt.subplots(1, len(alphas), figsize=(20, 5), sharey=True)
    for ax, a in zip(axes, alphas):
        recs = rec.recommend_for_user(uid, top_n=6, alpha=a, diversity_lambda=0.3)
        titles = [r["title"][:28] + ("…" if len(r["title"]) > 28 else "") for r in recs][::-1]
        scores = [r["score"] for r in recs][::-1]
        ax.barh(titles, scores, color="#4C72B0")
        label = "Pure Content" if a == 0.0 else ("Pure CF" if a == 1.0 else f"Hybrid α={a}")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, 1)
    fig.suptitle("How the CF/Content blend weight (alpha) reshapes recommendations", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {path}")


def plot_diversity_effect(rec, uid, filename="02_diversity_effect.png"):
    """Show how MMR diversity_lambda changes genre variety in the top-N list."""
    lambdas = [0.0, 0.3, 0.6]
    fig, axes = plt.subplots(1, len(lambdas), figsize=(18, 5))
    for ax, lam in zip(axes, lambdas):
        recs = rec.recommend_for_user(uid, top_n=8, alpha=0.6, diversity_lambda=lam)
        all_tags = []
        for r in recs:
            tags = str(r["genre_tags"]).split()
            all_tags.extend(tags[:2])  # top 2 tags per book
        tag_counts = pd.Series(all_tags).value_counts()
        ax.bar(tag_counts.index, tag_counts.values, color="#DD8452")
        ax.set_title(f"diversity_lambda = {lam}", fontsize=11)
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel("count in top-8 list")
    fig.suptitle("Genre-tag variety in recommendations as diversity weight increases", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_evaluation(results, filename="03_strategy_comparison.png"):
    strategies = list(results.keys())
    precision = [results[s]["precision@10"] for s in strategies]
    diversity = [results[s]["intra_list_diversity"] for s in strategies]
    coverage = [results[s]["catalog_coverage"] for s in strategies]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("Precision@10", precision, "#4C72B0"),
               ("Intra-list Diversity", diversity, "#55A868"),
               ("Catalog Coverage", coverage, "#C44E52")]
    for ax, (name, vals, color) in zip(axes, metrics):
        ax.bar(strategies, vals, color=color)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if name != "Catalog Coverage" else f"{v:.1%}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Hybrid vs Pure Collaborative Filtering vs Pure Content-Based", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main():
    rec = HybridBookRecommender()

    # Pick an active, interesting test user
    counts = rec.ratings["user_id"].value_counts()
    uid = counts.index[5]  # not the very top (bot-like) user, but still active
    user_ratings = rec.ratings[rec.ratings["user_id"] == uid].merge(
        rec.books[["book_id", "title"]], on="book_id")
    top5 = user_ratings.sort_values("rating", ascending=False).head(5)

    print("=" * 70)
    print(f"DEMO USER {uid} — top rated books:")
    for t in top5["title"]:
        print(f"   - {t}")

    hybrid_recs = rec.recommend_for_user(uid, top_n=8, alpha=0.6, diversity_lambda=0.3)
    print_recs("Hybrid recommendations (alpha=0.6, diversity_lambda=0.3)", hybrid_recs)

    cold_recs = rec.recommend_cold_start(
        ["The Hunger Games", "Harry Potter and the Sorcerer", "The Hobbit"], top_n=6)
    print_recs('Cold-start recommendations for someone who liked "The Hunger Games", '
               '"Harry Potter", "The Hobbit"', cold_recs)

    plot_score_breakdown(rec, uid)
    plot_diversity_effect(rec, uid)

    print("\nRunning evaluation (hybrid vs pure CF vs pure content)...")
    results = evaluate(top_n=10, n_test_users=150)
    plot_evaluation(results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for name, m in results.items():
        print(f"{name:32s} precision@10={m['precision@10']:.3f}  "
              f"diversity={m['intra_list_diversity']:.3f}  coverage={m['catalog_coverage']:.2%}")


if __name__ == "__main__":
    main()
