"""
Hybrid Book Recommender Engine
===============================

Combines two complementary signals:

  1. CONTENT-BASED  — TF-IDF over each book's title/author/genre-tags,
     compared via cosine similarity. Works even for brand-new users
     (cold start) since it needs no rating history.

  2. COLLABORATIVE FILTERING — Matrix factorization (Truncated SVD on the
     mean-centered user-item rating matrix) that learns latent taste
     factors from ~450K real reader ratings. Captures patterns content
     features can't see (e.g. "people who liked X also liked Y" even when
     X and Y are superficially very different books).

UNIQUE ANGLE — most tutorial hybrid recommenders just blend the two scores
and stop. This one adds:

  - Explainability: every recommendation reports which signal(s) drove it,
    which shared tags matched, and which of the user's own highly-rated
    books it resembles.
  - Diversity-aware re-ranking (Maximal Marginal Relevance): prevents the
    top-N list from being 10 near-duplicates of the same series/genre by
    penalizing candidates that are too similar to picks already made.
  - True cold-start mode: recommend for someone with zero rating history,
    using only a handful of books they say they liked.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


class HybridBookRecommender:
    def __init__(self, books_path="data/books_enriched.csv", ratings_path="data/ratings_sample.csv",
                 n_latent_factors=40, random_state=42):
        self.books = pd.read_csv(books_path)
        self.ratings = pd.read_csv(ratings_path)
        self.n_latent_factors = n_latent_factors
        self.random_state = random_state

        # book_id <-> matrix row index maps
        self.book_ids = self.books["book_id"].values
        self.book_id_to_idx = {bid: i for i, bid in enumerate(self.book_ids)}
        self.idx_to_book_id = {i: bid for bid, i in self.book_id_to_idx.items()}

        self._fit_content_model()
        self._fit_collaborative_model()

    # ------------------------------------------------------------------
    # CONTENT-BASED COMPONENT
    # ------------------------------------------------------------------
    def _fit_content_model(self):
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
        self.content_matrix = self.tfidf.fit_transform(self.books["content_text"].fillna(""))
        print(f"[content] TF-IDF matrix: {self.content_matrix.shape}")

    def content_similar_books(self, book_id, top_n=10, exclude=None):
        """Books most similar to a given book by content (title/author/genre)."""
        idx = self.book_id_to_idx[book_id]
        sims = cosine_similarity(self.content_matrix[idx], self.content_matrix).flatten()
        order = np.argsort(-sims)
        exclude = exclude or set()
        exclude = exclude | {idx}
        results = [(self.idx_to_book_id[i], sims[i]) for i in order if i not in exclude][:top_n]
        return results

    # ------------------------------------------------------------------
    # COLLABORATIVE FILTERING COMPONENT (matrix factorization via SVD)
    # ------------------------------------------------------------------
    def _fit_collaborative_model(self):
        users = self.ratings["user_id"].unique()
        self.user_id_to_idx = {uid: i for i, uid in enumerate(users)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}

        rows = self.ratings["user_id"].map(self.user_id_to_idx)
        cols = self.ratings["book_id"].map(self.book_id_to_idx)
        vals = self.ratings["rating"].astype(float)

        n_users, n_books = len(users), len(self.book_ids)
        R = sparse.coo_matrix((vals, (rows, cols)), shape=(n_users, n_books)).tocsr()
        self.raw_matrix = R

        # mean-center each user's ratings (only over observed entries) so SVD
        # models deviation-from-average-taste rather than raw rating scale
        user_means = np.asarray(R.sum(axis=1)).flatten() / np.maximum(np.asarray((R != 0).sum(axis=1)).flatten(), 1)
        self.user_means = user_means

        R_centered = R.tolil(copy=True)
        nz_rows, nz_cols = R.nonzero()
        R_centered[nz_rows, nz_cols] = np.asarray(R[nz_rows, nz_cols]).flatten() - user_means[nz_rows]
        R_centered = R_centered.tocsr()

        svd = TruncatedSVD(n_components=self.n_latent_factors, random_state=self.random_state)
        self.user_factors = svd.fit_transform(R_centered)          # (n_users, k)
        self.item_factors = svd.components_.T                     # (n_books, k)
        self.svd = svd

        print(f"[collaborative] user-item matrix: {R.shape}, "
              f"density={R.nnz / (n_users * n_books):.4%}, factors={self.n_latent_factors}")
        print(f"[collaborative] explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.3f}")

    def predicted_rating(self, user_id, book_id):
        u = self.user_id_to_idx.get(user_id)
        b = self.book_id_to_idx.get(book_id)
        if u is None or b is None:
            return None
        est = self.user_means[u] + self.user_factors[u].dot(self.item_factors[b])
        return float(np.clip(est, 1, 5))

    def cf_top_n(self, user_id, top_n=50, exclude_rated=True):
        """Top book candidates for a known user purely from the CF model."""
        u = self.user_id_to_idx.get(user_id)
        if u is None:
            return []
        scores = self.user_means[u] + self.item_factors.dot(self.user_factors[u])
        rated_idx = set()
        if exclude_rated:
            rated_idx = set(self.raw_matrix[u].nonzero()[1])
        order = np.argsort(-scores)
        results = [(self.idx_to_book_id[i], float(np.clip(scores[i], 1, 5)))
                   for i in order if i not in rated_idx][:top_n]
        return results

    # ------------------------------------------------------------------
    # USER TASTE PROFILE (for content-based scoring + cold start)
    # ------------------------------------------------------------------
    def user_content_profile(self, user_id=None, liked_book_ids=None, rating_threshold=4):
        """Build a single 'taste vector' in content-feature space, either from
        a known user's highly-rated books, or from an explicit liked-books list
        (cold start)."""
        if liked_book_ids is not None:
            idxs = [self.book_id_to_idx[b] for b in liked_book_ids if b in self.book_id_to_idx]
            weights = np.ones(len(idxs))
        else:
            user_ratings = self.ratings[(self.ratings["user_id"] == user_id) &
                                         (self.ratings["rating"] >= rating_threshold)]
            idxs = [self.book_id_to_idx[b] for b in user_ratings["book_id"] if b in self.book_id_to_idx]
            weights = user_ratings["rating"].values.astype(float) if len(idxs) else np.array([])

        if len(idxs) == 0:
            return None
        vecs = self.content_matrix[idxs]
        weights = weights / weights.sum()
        profile = vecs.T.dot(weights)
        return sparse.csr_matrix(profile).reshape(1, -1) if not sparse.issparse(profile) else profile.reshape(1, -1)

    # ------------------------------------------------------------------
    # HYBRID RECOMMENDATION WITH DIVERSITY (MMR) + EXPLANATIONS
    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id, top_n=10, alpha=0.6, diversity_lambda=0.3,
                            candidate_pool=100):
        """
        alpha: weight on collaborative-filtering score vs content score (0..1)
        diversity_lambda: 0 = pure relevance ranking, 1 = maximize diversity
        """
        is_known_user = user_id in self.user_id_to_idx

        if is_known_user:
            cf_candidates = self.cf_top_n(user_id, top_n=candidate_pool)
            candidate_ids = [b for b, _ in cf_candidates]
            cf_scores = {b: s for b, s in cf_candidates}
            profile = self.user_content_profile(user_id=user_id)
        else:
            raise ValueError("Unknown user_id — use recommend_cold_start() instead.")

        if profile is None or len(candidate_ids) == 0:
            # fall back to pure CF or pure popularity
            return self._popularity_fallback(top_n)

        cand_idxs = [self.book_id_to_idx[b] for b in candidate_ids]
        content_sims = cosine_similarity(profile, self.content_matrix[cand_idxs]).flatten()
        content_scores = {b: s for b, s in zip(candidate_ids, content_sims)}

        # normalize CF scores (1-5 scale) to 0-1 to combine fairly with cosine sim
        cf_vals = np.array([cf_scores[b] for b in candidate_ids])
        cf_norm = (cf_vals - 1) / 4.0
        content_vals = np.array([content_scores[b] for b in candidate_ids])

        hybrid_scores = {
            b: alpha * cf_n + (1 - alpha) * c
            for b, cf_n, c in zip(candidate_ids, cf_norm, content_vals)
        }

        ranked = self._mmr_rerank(candidate_ids, hybrid_scores, diversity_lambda, top_n)

        return self._build_explanations(ranked, user_id, cf_scores, content_scores, hybrid_scores,
                                         is_cold_start=False)

    def recommend_cold_start(self, liked_titles_or_ids, top_n=10, diversity_lambda=0.3,
                              candidate_pool=200):
        """Recommend for a brand-new user with no rating history, based on a
        few books they say they liked (pure content-based, since there's no
        rating signal to feed the CF model)."""
        liked_ids = self._resolve_ids(liked_titles_or_ids)
        profile = self.user_content_profile(liked_book_ids=liked_ids)
        if profile is None:
            raise ValueError("Could not match any of the provided books.")

        sims = cosine_similarity(profile, self.content_matrix).flatten()
        exclude = {self.book_id_to_idx[b] for b in liked_ids}
        order = np.argsort(-sims)
        candidates = [(self.idx_to_book_id[i], sims[i]) for i in order if i not in exclude][:candidate_pool]
        candidate_ids = [b for b, _ in candidates]
        content_scores = {b: s for b, s in candidates}

        ranked = self._mmr_rerank(candidate_ids, content_scores, diversity_lambda, top_n)
        return self._build_explanations(ranked, None, {}, content_scores, content_scores,
                                         is_cold_start=True, seed_books=liked_ids)

    def _resolve_ids(self, titles_or_ids):
        resolved = []
        for x in titles_or_ids:
            if isinstance(x, (int, np.integer)) and x in self.book_id_to_idx:
                resolved.append(x)
            else:
                match = self.books[self.books["title"].str.contains(str(x), case=False, na=False, regex=False)]
                if len(match):
                    resolved.append(int(match.iloc[0]["book_id"]))
        return resolved

    def _popularity_fallback(self, top_n):
        top = self.books.sort_values(["average_rating", "ratings_count"], ascending=False).head(top_n)
        return [{"book_id": int(r.book_id), "title": r.title, "authors": r.authors,
                 "score": float(r.average_rating), "why": "Popular highly-rated book (fallback)"}
                for r in top.itertuples()]

    # ------------------------------------------------------------------
    # DIVERSITY-AWARE RE-RANKING (Maximal Marginal Relevance)
    # ------------------------------------------------------------------
    def _mmr_rerank(self, candidate_ids, relevance_scores, diversity_lambda, top_n):
        if not candidate_ids:
            return []
        cand_idxs = [self.book_id_to_idx[b] for b in candidate_ids]
        sim_matrix = cosine_similarity(self.content_matrix[cand_idxs])

        selected = []
        remaining = list(range(len(candidate_ids)))

        # seed with the highest-relevance candidate
        first = max(remaining, key=lambda i: relevance_scores[candidate_ids[i]])
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < top_n:
            def mmr_score(i):
                relevance = relevance_scores[candidate_ids[i]]
                redundancy = max(sim_matrix[i][j] for j in selected)
                return (1 - diversity_lambda) * relevance - diversity_lambda * redundancy

            nxt = max(remaining, key=mmr_score)
            selected.append(nxt)
            remaining.remove(nxt)

        return [candidate_ids[i] for i in selected]

    # ------------------------------------------------------------------
    # EXPLANATIONS
    # ------------------------------------------------------------------
    def _build_explanations(self, ranked_book_ids, user_id, cf_scores, content_scores,
                             hybrid_scores, is_cold_start, seed_books=None):
        results = []
        user_top_books = []
        if user_id is not None:
            ur = self.ratings[(self.ratings["user_id"] == user_id)].sort_values("rating", ascending=False)
            user_top_books = [self.book_id_to_idx[b] for b in ur["book_id"].head(20) if b in self.book_id_to_idx]

        for bid in ranked_book_ids:
            row = self.books.loc[self.books["book_id"] == bid].iloc[0]
            idx = self.book_id_to_idx[bid]
            reasons = []

            if is_cold_start:
                reasons.append(f"similar in genre/author to books you liked (score={content_scores.get(bid, 0):.2f})")
            else:
                cf = cf_scores.get(bid)
                cs = content_scores.get(bid)
                if cf is not None:
                    reasons.append(f"readers with similar taste rated it ~{cf:.1f}/5 (collaborative signal)")
                if cs is not None:
                    reasons.append(f"matches your genre/author preferences (content similarity={cs:.2f})")

                if user_top_books:
                    sims_to_favs = cosine_similarity(self.content_matrix[idx], self.content_matrix[user_top_books]).flatten()
                    best = np.argmax(sims_to_favs)
                    if sims_to_favs[best] > 0.15:
                        fav_title = self.books.loc[self.books["book_id"] ==
                                                    self.idx_to_book_id[user_top_books[best]], "title"].values[0]
                        reasons.append(f"resembles a book you rated highly: \"{fav_title}\"")

            results.append({
                "book_id": int(bid),
                "title": row["title"],
                "authors": row["authors"],
                "genre_tags": row["genre_tags"],
                "avg_rating": float(row["average_rating"]),
                "score": float(hybrid_scores.get(bid, content_scores.get(bid, 0))),
                "why": "; ".join(reasons) if reasons else "matched your profile",
            })
        return results
