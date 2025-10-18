"""
Recommendation script.
- Parses user input.
- Load trained model.
- Generate recommendations.
"""

import argparse
import os
import joblib
import pandas as pd

ARTIFACT_DIR = os.path.join("artifacts")
KNN_PATH = os.path.join(ARTIFACT_DIR, "knn.joblib")
FEATS_PATH = os.path.join(ARTIFACT_DIR, "features_df.parquet")

def main():
    parser = argparse.ArgumentParser(description="Recommend similar book titles.")
    parser.add_argument("--title", required=True, help="Book title to find neighbours for.")
    parser.add_argument("--top-k", type=int, default=5, help="How many similar titles to show.")
    args = parser.parse_args()

    # Make sure artifacts exist.
    if not os.path.exists(KNN_PATH) or not os.path.exists(FEATS_PATH):
        raise SystemExit("Artifacts not found. Run `python -m scripts.train` first.")

    # Load trained model.
    print("Loading artifacts...")
    knn = joblib.load(KNN_PATH)
    features_df = pd.read_parquet(FEATS_PATH)

    # Create an instance of the recommender
    from bookrec.recommender import KNNRecommender
    rec = KNNRecommender(features_df=features_df, model=knn)

    # Generate some recommendations
    base, neighbours = rec.recommend(args.title, top_k=args.top_k)
    print(f"Base title: {base}")
    for i, (t, d) in enumerate(neighbours, 1):
        print(f"{i}) {t} (distance: {d:.2f})")

if __name__ == "__main__":
    main()
