"""
Training script for the book recommender.
- Fetches book-crossing data.
- Filters to active users & popular books.
- Builds item x user ratings matrix.
- Trains a k-NN model with cosine distance.
- Evaluates with a Recall@K metric using user-level holdout.
- Saves the model.
"""

import argparse
import os
import joblib

from bookrec.data import download_bookcrossing, load_books, load_ratings
from bookrec.pipeline import filter_active_popular, build_item_user_matrix, train_knn
from bookrec.evaluation import train_test_holdout_per_user, recall_at_k_from_item_neighbours

# Ensure artifacts/ directory exists for output
ARTIFACT_DIR = os.path.join("artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Train KNN book recommender.")
    # Where the zip will be stored
    parser.add_argument("--data-dir", default="data", help="Directory to download data.")
    # Thresholds
    parser.add_argument("--min-user-ratings", type=int, default=200)
    parser.add_argument("--min-book-ratings", type=int, default=100)
    # How many neighbours to use
    parser.add_argument("--k", type=int, default=5, help="Top-k neighbours to consider for evaluation.")
    # Cut off for positive ratings (e.g. >= 7 out of 10)
    parser.add_argument("--positive-threshold", type=float, default=7.0)
    args = parser.parse_args()

    print("Downloading data (if required)...")
    books_csv, ratings_csv = download_bookcrossing(args.data_dir)

    # Create metadata & rating CSVs
    print("Loading CSVs...")
    books = load_books(books_csv)
    ratings = load_ratings(ratings_csv)

    print("Filtering active users and popular books...")
    ratings_f = filter_active_popular(ratings, args.min_user_ratings, args.min_book_ratings)

    # Create item x user matrix of ratings.
    # features_df is our matrix
    # features_csr has the same values as features_df, but only stores non-zeros
    # We use csr with scikit-learn
    print("Building item-user matrix (titles x users)...")
    features_df, features_csr = build_item_user_matrix(ratings_f, books)

    # Train the k-NN model using cosine distance.
    print("Training k-NN model...")
    knn = train_knn(features_csr, n_neighbours=max(args.k + 1, 6))

    # Recall@K with per-user positive holdout
    # Basically evaluates how well the algorithm is performing
    print("Evaluating...")
    train_df, test_df = train_test_holdout_per_user(ratings_f, positive_threshold=args.positive_threshold)
    recall = recall_at_k_from_item_neighbours(
        features_df=features_df,
        knn=knn,
        books=books,
        train_df=train_df,
        test_df=test_df,
        k=args.k,
        positive_threshold=args.positive_threshold,
    )
    print(f"Recall@{args.k}: {recall:.3f}")

    # Save artifacts
    print("Saving artifacts...")
    joblib.dump(knn, os.path.join(ARTIFACT_DIR, "knn.joblib"))
    features_df.to_parquet(os.path.join(ARTIFACT_DIR, "features_df.parquet"))
    print("Done.")

if __name__ == "__main__":
    main()
