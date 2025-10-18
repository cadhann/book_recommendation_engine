"""
Evaluation for the pipeline.
"""

from typing import Dict, List, Tuple
import random
import numpy as np
import pandas as pd

def train_test_holdout_per_user(
    ratings: pd.DataFrame, positive_threshold: float = 7.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the full ratings dataframe into two parts:
    - train_df: almost all user-book ratings
    - test_df: test set containing 1 positive rating per user with >= 2
               positive ratings
    ratings: ratings pandas dataframe
    positive_threshold: minimum value of a "positive" rating.
    Returns the training and testing sets as a pandas dataframe tuple.
    """
    r = ratings.copy()
    r["positive"] = r["rating"] >= positive_threshold

    # Prepare mask for train rows
    train_mask = pd.Series(True, index=r.index)

    # For each user, check if they have more than 2 positive ratings.
    # If they do, randomly select one positive to hold out.
    # Mark as false in train_mask so not included in training data.
    test_rows = []
    for user, grp in r.groupby("user"):
        pos = grp[grp["positive"]]
        if len(pos) >= 2:
            # Keep original label index
            holdout_idx = pos.sample(1, random_state=42).index[0]
            test_rows.append(r.loc[holdout_idx, ["user", "isbn", "rating"]])
            train_mask.loc[holdout_idx] = False

    # Build train & test dataframes.
    train_df = r.loc[train_mask, ["user", "isbn", "rating"]]
    test_df = (
        pd.DataFrame(test_rows, columns=["user", "isbn", "rating"])
        if test_rows else
        pd.DataFrame(columns=["user", "isbn", "rating"])
    )

    return train_df, test_df

def recall_at_k_from_item_neighbours(
    features_df: pd.DataFrame,
    knn, #NearestNeighbours
    books: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 5,
    positive_threshold: float = 7.0,
) -> float:
    """
    Calculate Recall@K.
    features_df: the item x user matrix
    knn: NearestNeighbours corresponding to features_df
    books: book metadata in pandas dataframe
    train_df: train set
    test_df: test set
    k: how many neighbours per positive
    positive_threshold: rating cutoff for "positive" ratings.
    Returns the result of Recall@K
    """
    # map isbn -> title
    book_titles = books.set_index("isbn")["title"].to_dict()

    # Get the positive values per user in training set
    train_positive = train_df[train_df["rating"] >= positive_threshold]
    user_pos_titles = (
        train_positive
        .merge(books[["isbn", "title"]], on="isbn", how="left")
        .groupby("user")["title"].apply(list).to_dict()
    )


    hits = 0
    total = 0

    # Iterate through each test row
    for _, row in test_df.iterrows():
        # Get user and the hidden book.
        user = row["user"]
        isbn = row["isbn"]
        heldout_title = book_titles.get(isbn)
        # Skip if not enough data
        if not heldout_title or user not in user_pos_titles:
            continue

        # Get user's known "liked" books
        base_titles = [t for t in user_pos_titles[user] if t in features_df.index]
        if not base_titles:
            continue

        # Find neighbours for everything they liked.
        neighbour_titles = set()
        for t in base_titles:
            distances, indices = knn.kneighbors(features_df.loc[t].values.reshape(1, -1))
            neighbours = [features_df.index[i] for i in indices.flatten()[1:k+1]]
            neighbour_titles.update(neighbours)

        # if hidden book is in the list, count a hit.
        total += 1
        if heldout_title in neighbour_titles:
            hits += 1

    if total == 0:
        return 0.0
    return hits / total
