"""
Fetches and prepares the raw dataset for the recommender.
"""

import os
import io
import zipfile
import requests
import pandas as pd
from typing import Tuple

BOOKS_URL = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"
BOOKS_CSV = "BX-Books.csv"
RATINGS_CSV = "BX-Book-Ratings.csv"

def download_bookcrossing(data_dir: str) -> Tuple[str, str]:
    """
    Download and extract the Book-Crossing dataset.
    data_dir: the directory to store data.
    Returns (books_csv_path, ratings_csv_path).
    """

    # Ensure data/ directory exists
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "book-crossings.zip")

    # If the dataset isn't there, download it
    if not os.path.exists(zip_path):
        resp = requests.get(BOOKS_URL, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)

    # Extract zip
    # This should contain BX-Books.csv and BX-Book-Ratings.csv
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    # Verify files exist
    books_path = os.path.join(data_dir, BOOKS_CSV)
    ratings_path = os.path.join(data_dir, RATINGS_CSV)
    if not os.path.exists(books_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError("Expected CSVs not found after extraction.")

    return books_path, ratings_path


def load_books(books_csv: str) -> pd.DataFrame:
    """
    Load books: isbn, title, author (ISO-8859-1, ';' separated)
    books_csv: the csv path for the books.
    Returns a pandas dataframe of the books
    """
    try:
        # Attempt to load data
        return pd.read_csv(
            books_csv,
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=["isbn", "title", "author"],
            usecols=["isbn", "title", "author"],
            dtype={"isbn": "str", "title": "str", "author": "str"},
            engine="python",
            on_bad_lines="error", # raise error if rows are malformed
        )
    except Exception:
        # Drop any malformed rows
        return pd.read_csv(
            books_csv,
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=["isbn", "title", "author"],
            usecols=["isbn", "title", "author"],
            dtype={"isbn": "str", "title": "str", "author": "str"},
            engine="python",
            on_bad_lines="skip", # drop malformed rows
        )
    except Exception:
        # If that still doesn't work, ignore weird quotes
        return pd.read_csv(
            books_csv,
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=["isbn", "title", "author"],
            usecols=["isbn", "title", "author"],
            dtype={"isbn": "str", "title": "str", "author": "str"},
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE, # ignore quoting
            escapechar="\\",
        )


def load_ratings(ratings_csv: str) -> pd.DataFrame:
    """
    Load ratings: user, isbn, rating (ISO-8859-1, ';' separated).
    ratings_csv: csv path to the ratings
    Returns a pandas dataframe of the ratings.
    """
    try:
        # Attempt to load data
        return pd.read_csv(
            ratings_csv,
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=["user", "isbn", "rating"],
            usecols=["user", "isbn", "rating"],
            dtype={"user": "int32", "isbn": "str", "rating": "float32"},
            engine="python",
            on_bad_lines="error",
        )
    except Exception:
        # If there are malformed rows, skip them
        return pd.read_csv(
            ratings_csv,
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=["user", "isbn", "rating"],
            usecols=["user", "isbn", "rating"],
            dtype={"user": "int32", "isbn": "str", "rating": "float32"},
            engine="python",
            on_bad_lines="skip", # drop malformed rows
        )
