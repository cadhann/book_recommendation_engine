"""
Book recommender.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches

@dataclass
class KNNRecommender:
    features_df: pd.DataFrame
    model: NearestNeighbors

    @property
    def titles(self) -> List[str]:
        """
        Returns list of all the book titles.
        """
        return list(self.features_df.index)

    def _find_title_index(self, title: str) -> Optional[int]:
        """
        Take's user input and tries to figure out which row it refers to.
        title: user's input.
        Returns the row number if it finds it.
        """
        # Try look for an exact match first
        if title in self.features_df.index:
            return self.features_df.index.get_loc(title)
        # If that doesn't work, try for a case-insensitive match
        lower_map = {t.lower(): t for t in self.features_df.index}
        if title.lower() in lower_map:
            canonical = lower_map[title.lower()]
            return self.features_df.index.get_loc(canonical)
        # Find the closest match (within 60%)
        candidates = get_close_matches(title, self.titles, n=1, cutoff=0.6)
        if candidates:
            return self.features_df.index.get_loc(candidates[0])
        # If we still can't find it, give up
        return None

    def recommend(self, title: str, top_k: int = 5) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Generates recommendations.
        title: base title
        top_k: the top k neighbours
        Return (base_title, [(neighbour_title, distance), ...]) with cosine distances.
        """

        # Find book's row index.
        idx = self._find_title_index(title)
        if idx is None:
            raise ValueError(f"Title '{title}' not found in index.")

        # Get nearest neighbours
        distances, indices = self.model.kneighbors(self.features_df.iloc[idx, :].values.reshape(1, -1))
        distances = distances.flatten().tolist()
        indices = indices.flatten().tolist()

        # Skip the first neighbour as it's always itself.
        pairs = []
        for d, i in zip(distances[1: top_k + 1], indices[1: top_k + 1]):
            pairs.append((self.features_df.index[i], float(d)))
        return (self.features_df.index[idx], pairs)
