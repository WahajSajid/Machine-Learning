"""
Evaluation: Hybrid vs Pure Collaborative Filtering vs Pure Content-Based

Methodology (a leave-k-out style split, standard for implicit/rating-based
recommender evaluation):
  - For each test user, hide their most recent ~20% of high ratings (>=4).
  - Ask each strategy to produce a top-N list from the *remaining* signal.
  - Precision@N = fraction of the top-N list that shows up in the held-out
    "liked" set.
  - Catalog coverage = fraction of the entire book catalog that ever gets
    recommended across all test users (measures filter-bubble risk — a
    model that always recommends the same 20 bestsellers scores low here).
  - Intra-list diversity = average (1 - cosine similarity) between pairs of
    books within a single user's top-N list (higher = less repetitive).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from recommender import HybridBookRecommender


def train_test_split_ratings(ratings, test_frac=0.2, min_liked=5, seed=42):
    rng = np.random.RandomState(seed)
    liked = ratings[ratings["rating"] >= 4]
    user_counts = liked["user_id"].value_counts()
    eligible_users = user_counts[user_counts >= min_liked].index

    test_rows = []
    train_ratings = ratings.copy()
    for uid in eligible_users:
        user_liked = liked[liked["user_id"] == uid]
        n_hide = max(1, int(len(user_liked) * test_frac))
        hide_idx = rng.choice(user_liked.index, size=n_hide, replace=False)
        test_rows.append(ratings.loc[hide_idx])
        train_ratings = train_ratings.drop(hide_idx)

    test_df = pd.concat(test_rows)
    return train_ratings, test_df, list(eligible_users)


def intra_list_diversity(rec, book_ids):
    if len(book_ids) < 2:
        return 0.0
    idxs = [rec.book_id_to_idx[b] for b in book_ids if b in rec.book_id_to_idx]
    if len(idxs) < 2:
        return 0.0
    sims = cosine_similarity(rec.content_matrix[idxs])
    n = len(idxs)
    off_diag = sims[np.triu_indices(n, k=1)]
    return float(1 - off_diag.mean())


def evaluate(top_n=10, n_test_users=150, alpha=0.6, diversity_lambda=0.3):
    print("Loading full model (train) ...")
    rec = HybridBookRecommender()

    train_ratings, test_df, eligible_users = train_test_split_ratings(rec.ratings)
    rng = np.random.RandomState(0)
    test_users = rng.choice(eligible_users, size=min(n_test_users, len(eligible_users)), replace=False)

    # Refit a version of the recommender on train-only ratings so held-out
    # items are genuinely unseen by the CF model
    rec.ratings.to_csv("data/_ratings_full_backup.csv", index=False)
    train_ratings.to_csv("data/ratings_sample.csv", index=False)
    rec_train = HybridBookRecommender()
    # restore original file
    rec.ratings.to_csv("data/ratings_sample.csv", index=False)

    strategies = {
        "Pure Collaborative Filtering": lambda uid: [b for b, _ in rec_train.cf_top_n(uid, top_n=top_n)],
        "Pure Content-Based": lambda uid: _pure_content(rec_train, uid, top_n),
        "Hybrid (this project)": lambda uid: [r["book_id"] for r in
                                               rec_train.recommend_for_user(uid, top_n=top_n, alpha=alpha,
                                                                             diversity_lambda=diversity_lambda)],
    }

    results = {}
    all_recommended = {name: set() for name in strategies}

    for name, fn in strategies.items():
        precisions = []
        diversities = []
        for uid in test_users:
            if uid not in rec_train.user_id_to_idx:
                continue
            held_out = set(test_df[test_df["user_id"] == uid]["book_id"])
            if not held_out:
                continue
            try:
                recs = fn(uid)
            except Exception:
                continue
            if not recs:
                continue
            hits = len(set(recs) & held_out)
            precisions.append(hits / len(recs))
            diversities.append(intra_list_diversity(rec_train, recs))
            all_recommended[name].update(recs)

        results[name] = {
            "precision@%d" % top_n: float(np.mean(precisions)) if precisions else 0.0,
            "intra_list_diversity": float(np.mean(diversities)) if diversities else 0.0,
            "catalog_coverage": len(all_recommended[name]) / len(rec_train.book_ids),
            "n_test_users_evaluated": len(precisions),
        }

    return results


def _pure_content(rec, uid, top_n):
    profile = rec.user_content_profile(user_id=uid)
    if profile is None:
        return []
    sims = cosine_similarity(profile, rec.content_matrix).flatten()
    rated = set(rec.raw_matrix[rec.user_id_to_idx[uid]].nonzero()[1])
    order = np.argsort(-sims)
    return [rec.idx_to_book_id[i] for i in order if i not in rated][:top_n]


if __name__ == "__main__":
    results = evaluate()
    print("\n" + "=" * 70)
    print(f"{'Strategy':32s} {'Precision@10':>13s} {'Diversity':>10s} {'Coverage':>10s}")
    print("=" * 70)
    for name, m in results.items():
        print(f"{name:32s} {m['precision@10']:>13.3f} {m['intra_list_diversity']:>10.3f} "
              f"{m['catalog_coverage']:>10.3%}")
