# Hybrid Book Recommender — with Explanations & Diversity Control

A book recommendation system built on **real reader data** (the
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset: 10,000
books, ~450K sampled ratings from ~4,000 readers) that combines two
classic techniques and adds a few things most tutorial recommenders skip.

## What makes this one different

Most hybrid-recommender tutorials just blend a collaborative-filtering
score with a content-similarity score and stop there. This project adds:

1. **Explainability** — every recommendation says *why* it was picked:
   which collaborative signal drove it, which tags/genres matched, and
   which specific book from your own history it resembles.
2. **Diversity-aware re-ranking (MMR)** — a "maximal marginal relevance"
   pass prevents your top-10 list from being 10 near-identical sequels of
   the same series. A tunable `diversity_lambda` controls how much variety
   you want vs. raw relevance.
3. **Real cold start** — recommend for a user with *zero* rating history
   using nothing but a few book titles they say they liked (pure
   content-based fallback, no synthetic ratings required).
4. **A real evaluation, not just a demo** — precision@10, intra-list
   diversity, and catalog coverage are measured and compared across pure
   collaborative filtering, pure content-based, and the hybrid, on a
   held-out split of real ratings.

## How it works

### 1. Content-based component
Each book gets a text profile from its title, author, and curated
genre-like tags (folksonomy tags filtered down to ~70 real genre/theme
keywords, since raw Goodreads tags are mostly noise like "to-read" or
"owned"). TF-IDF + cosine similarity finds books that are textually/
thematically alike.

### 2. Collaborative filtering component
Ratings are assembled into a sparse user × book matrix, mean-centered per
user, and factorized with Truncated SVD (40 latent factors) — a standard,
fast matrix-factorization approach to collaborative filtering that
captures "readers like you also loved this" patterns the content model
can't see.

### 3. Hybrid scoring
```
hybrid_score = alpha * normalized_CF_score + (1 - alpha) * content_similarity
```
`alpha` is tunable — `alpha=1` is pure collaborative filtering, `alpha=0`
is pure content-based.

### 4. MMR diversity re-ranking
Candidates are greedily selected to maximize
`(1 - diversity_lambda) * relevance - diversity_lambda * max_similarity_to_already_picked`,
trading off relevance against redundancy.

## Evaluation results

On a held-out split of 150 test users (most recent 20% of each user's
4-5 star ratings hidden, then re-predicted):

| Strategy | Precision@10 | Intra-list Diversity | Catalog Coverage |
|---|---|---|---|
| Pure Collaborative Filtering | ~0.098 | 0.858 | 1.7% |
| Pure Content-Based | ~0.097 | 0.401 | 7.4% |
| **Hybrid (this project)** | **~0.115** | **0.876** | 4.1% |

The hybrid beats both pure approaches on precision *and* diversity —
because CF and content-based models make different mistakes, blending
them cancels out some of each one's blind spots. (Exact numbers vary
slightly run to run due to random sampling of test users.)

## Project structure

```
book_recommender/
├── data/
│   ├── books_enriched.csv         # cleaned books + curated genre tags (included, ready to use)
│   └── ratings_sample.csv         # filtered/trimmed ratings, ~450K real ratings (included)
├── prepare_data.py                # regenerates the two files above from the raw goodbooks-10k
│                                   # CSVs (books.csv, ratings.csv, tags.csv, book_tags.csv) —
│                                   # only needed if you want to re-derive them; download the raw
│                                   # files from https://github.com/zygmuntz/goodbooks-10k first
├── recommender.py                 # HybridBookRecommender engine (the core of the project)
├── evaluate.py                    # precision@k / diversity / coverage comparison
├── demo.py                        # runs example recommendations + saves charts
├── requirements.txt
└── outputs/                       # generated after running demo.py
    ├── 01_score_breakdown.png
    ├── 02_diversity_effect.png
    └── 03_strategy_comparison.png
```

## Running it

```bash
pip install -r requirements.txt
python demo.py
```

`data/books_enriched.csv` and `data/ratings_sample.csv` are already
included, so this runs immediately — no download needed. `prepare_data.py`
is only needed if you want to regenerate those files from the raw
goodbooks-10k CSVs (e.g. to change the filtering thresholds or use the
full 6M-row ratings file instead of the trimmed sample).

## Using it in your own code

```python
from recommender import HybridBookRecommender

rec = HybridBookRecommender()

# Recommendations for an existing user (blends CF + content, diversified)
recs = rec.recommend_for_user(user_id=12345, top_n=10, alpha=0.6, diversity_lambda=0.3)

# Recommendations for someone brand new, based on books they say they liked
recs = rec.recommend_cold_start(["The Hobbit", "Dune"], top_n=10)

for r in recs:
    print(r["title"], "-", r["why"])
```

## Ideas to extend this project

- Swap Truncated SVD for a proper implicit-feedback model (e.g. ALS via
  the `implicit` library) if you move to click/purchase data instead of
  explicit 1-5 star ratings.
- Add a Streamlit UI: a search box for "books I've liked" + a slider for
  `alpha` and `diversity_lambda` so users can tune their own feed live.
- Incorporate publication year / recency into content features to support
  "surprise me with something new" vs "more of the same" modes.
- Try a learned hybrid weight (a small model that predicts the best
  `alpha` per user based on how much rating history they have) instead of
  a single global `alpha`.
