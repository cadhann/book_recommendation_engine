"""
Core training pipeline.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def filter_active_popular(
    ratings: pd.DataFrame,
    min_user_ratings: int = 200,
    min_book_ratings: int = 100,
) -> pd.DataFrame:
    """
    Keep only active users and popular books to reduce size &
    quality of dataset.
    ratings: the full (user, isbn, rating) dataframe from load_ratings()
    min_user_ratings: ratings a user must have to be "active". (default: 200)
    min_book_ratings: ratings a book must have to be "popular". (default: 100)
    Returns a filtered dataset matching the conditions above.
    """
    user_counts = ratings["user"].value_counts()
    book_counts = ratings["isbn"].value_counts()
    filtered = ratings[
        ratings["user"].isin(user_counts[user_counts >= min_user_ratings].index)
        & ratings["isbn"].isin(book_counts[book_counts >= min_book_ratings].index)
    ].copy()
    return filtered

def build_item_user_matrix(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    positive_threshold: float = 7.0,
    tfidf: bool = True,
) -> Tuple[pd.DataFrame, csr_matrix]:
    """
    Build the item x user matrix using implicit positives
    and optionally applies TF-IDF style up/downweighting per user.
    ratings: pandas dataframe containing ratings
    books: pandas dataframe containing books
    positive_threshold: minimum rating value for a rating to be considered
                        "positive".
    tfidf: whether or not to use TF-IDF weighting
    Returns the matrix as a dataframe and csr in a tuple.
    """

    # Join raw ratings with book metadata.
    df = pd.merge(ratings, books, on="isbn", how="inner")

    # Binarise ratings to either liked or not liked based on positive_threshold
    # 1 for liked (rating above threshold), 0 otherwise.
    df["bin"] = (df["rating"] >= positive_threshold).astype(np.float32)

    # Create matrix:
    # Rows = title
    # Columns = user
    # Cells = 1.0 if user "liked" the book, otherwise 0.0.
    # TODO: categorising by ISBN may improve results, would need to rewrite
    #       input arguments in train.py
    features_df = df.pivot_table(values="bin", index="title", columns="user", fill_value=0.0)

    X = features_df.values.astype(np.float32)

    # TF-IDF lets us remove some popularity bias.
    # Users who like everything are downweighted.
    # Users who don't like everything are upweighted.
    if tfidf:
        # IDF per user: log(N_items / df), df is the num.
        # items the user liked

        # Count how many books each user liked.
        dfu = (X > 0).sum(axis=0).clip(min=1)

        # Calculate IDF for each user.
        # Small dfu -> weight high
        # Large dfu -> weight low
        idf = np.log(X.shape[0] / dfu, dtype=np.float32)

        # Apply weights to the matrix
        X = X * idf  # broadcast user-wise weights

    # Generate CSR matrix for scikit-learn NearestNeighbors.
    features_csr = csr_matrix(X)
    return features_df, features_csr

def train_knn(
    features_csr: csr_matrix,
    metric: str = "cosine",
    algorithm: str = "brute",
    n_neighbours: int = 6,
) -> NearestNeighbors:
    """
    Fit a NearestNeighbors model on the matrix.
    features_csr: matrix from build_item_user_matrix()
    metric: similarity metric to use (default: cosine)
    algorithm: how neighbours are searched (default: brute)
    n_neighbours: number of neighbours to store when querying.
    Returns the result of the NearestNeighbors model.
    """
    knn = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_neighbours)
    knn.fit(features_csr)
    return knn
