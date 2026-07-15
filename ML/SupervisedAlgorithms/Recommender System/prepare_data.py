"""
Data preparation for the Hybrid Book Recommender project.

Source: goodbooks-10k dataset (https://github.com/zygmuntz/goodbooks-10k)
        10,000 books, ~6M ratings, folksonomy tags.

This script:
  1. Cleans book metadata and attaches human-readable "genre-like" tags
     to each book (filtered from the noisy folksonomy tag set).
  2. Filters ratings down to an active, well-rated core (removes very
     sparse users/books) and subsamples users so the shipped dataset
     stays small (~a few MB) while remaining real, non-synthetic data.

Run once to (re)generate data/books_enriched.csv and data/ratings_sample.csv.
The rest of the project reads only those two trimmed files, so it does not
need internet access after this step.
"""

import pandas as pd
import numpy as np
import re

RAW_DIR = "data"
SEED = 42
np.random.seed(SEED)

# A curated whitelist of tag substrings that actually look like genres/themes,
# to filter out noise like "to-read", "owned", "favorites", "kindle", etc.
GENRE_KEYWORDS = [
    "fiction", "fantasy", "romance", "mystery", "thriller", "horror",
    "science-fiction", "sci-fi", "dystopia", "young-adult", "historical",
    "biography", "memoir", "nonfiction", "non-fiction", "classic",
    "poetry", "graphic-novel", "comics", "adventure", "crime", "war",
    "philosophy", "psychology", "self-help", "business", "history",
    "religion", "spirituality", "humor", "short-stories", "literary",
    "contemporary", "paranormal", "steampunk", "cyberpunk", "western",
    "childrens", "middle-grade", "novella", "dark", "gothic", "epic",
    "magic", "vampires", "zombies", "dragons", "space", "time-travel",
    "politics", "science", "true-crime", "sports", "cooking", "travel",
    "art", "music", "sociology", "economics", "feminism", "lgbt",
]


def clean_genre_tags(books, book_tags, tags):
    tag_lookup = tags.set_index("tag_id")["tag_name"].to_dict()
    book_tags = book_tags.copy()
    book_tags["tag_name"] = book_tags["tag_id"].map(tag_lookup)
    book_tags = book_tags.dropna(subset=["tag_name"])

    pattern = re.compile("|".join(GENRE_KEYWORDS))
    book_tags = book_tags[book_tags["tag_name"].str.lower().str.contains(pattern)]

    # Keep the top N genre tags per book by tag "count" (popularity of that tag
    # among readers who shelved the book)
    book_tags = book_tags.sort_values("count", ascending=False)
    top_tags = (
        book_tags.groupby("goodreads_book_id")["tag_name"]
        .apply(lambda s: " ".join(s.head(6)))
        .reset_index()
        .rename(columns={"tag_name": "genre_tags"})
    )

    books = books.merge(top_tags, on="goodreads_book_id", how="left")
    books["genre_tags"] = books["genre_tags"].fillna("")
    return books


def build_content_text(books):
    def row_text(row):
        author = str(row["authors"]).replace(",", " ").replace("-", " ")
        title = str(row["title"])
        tags = str(row["genre_tags"]).replace("-", " ")
        return f"{title} {author} {author} {tags} {tags} {tags}"  # weight author/tags more

    books["content_text"] = books.apply(row_text, axis=1)
    return books


def prepare_books():
    books = pd.read_csv(f"{RAW_DIR}/books.csv")
    book_tags = pd.read_csv(f"{RAW_DIR}/book_tags.csv")
    tags = pd.read_csv(f"{RAW_DIR}/tags.csv")

    books = clean_genre_tags(books, book_tags, tags)
    books = build_content_text(books)

    keep_cols = [
        "book_id", "goodreads_book_id", "authors", "original_publication_year",
        "title", "average_rating", "ratings_count", "genre_tags", "content_text",
    ]
    books = books[keep_cols]
    books.to_csv(f"{RAW_DIR}/books_enriched.csv", index=False)
    print(f"Saved books_enriched.csv: {books.shape}")
    return books


def prepare_ratings(min_user_ratings=20, max_user_ratings=200,
                     min_book_ratings=50, n_users_sample=4000):
    ratings = pd.read_csv(f"{RAW_DIR}/ratings.csv")
    print(f"Raw ratings: {ratings.shape}")

    # Drop duplicate (user, book) pairs, keep first
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"])

    # Filter books with enough ratings to be meaningful
    book_counts = ratings["book_id"].value_counts()
    good_books = book_counts[book_counts >= min_book_ratings].index
    ratings = ratings[ratings["book_id"].isin(good_books)]

    # Filter to moderately active users (avoid one-rating users and mega-bot users)
    user_counts = ratings["user_id"].value_counts()
    good_users = user_counts[(user_counts >= min_user_ratings) & (user_counts <= max_user_ratings)].index
    ratings = ratings[ratings["user_id"].isin(good_users)]

    # Subsample users to keep the shipped dataset small
    sampled_users = np.random.choice(ratings["user_id"].unique(),
                                      size=min(n_users_sample, ratings["user_id"].nunique()),
                                      replace=False)
    ratings = ratings[ratings["user_id"].isin(sampled_users)]

    ratings.to_csv(f"{RAW_DIR}/ratings_sample.csv", index=False)
    print(f"Saved ratings_sample.csv: {ratings.shape}")
    print(f"  Unique users: {ratings['user_id'].nunique()}, unique books: {ratings['book_id'].nunique()}")
    return ratings


if __name__ == "__main__":
    prepare_books()
    prepare_ratings()
