# src/models/popularity.py
import pandas as pd

class PopularityRecommender:
    """
    Simple baseline: rank items by a popularity score.
    score = average_rating * sqrt(number_of_ratings)
    """
    def __init__(self, min_ratings: int = 1):
        self.min_ratings = min_ratings
        self.rankings = None

    def fit(self, ratings: pd.DataFrame):
        # ratings must have columns: userId, productId, Rating
        g = ratings.groupby("productId", as_index=False).agg(
            avg_rating=("Rating", "mean"),
            n=("Rating", "count")
        )
        g = g[g["n"] >= self.min_ratings].copy()
        g["score"] = g["avg_rating"] * (g["n"] ** 0.5)
        self.rankings = g.sort_values("score", ascending=False).reset_index(drop=True)
        return self

    def top_k(self, k: int = 10) -> pd.DataFrame:
        if self.rankings is None:
            raise RuntimeError("Call fit() first.")
        return self.rankings.loc[:, ["productId", "avg_rating", "n", "score"]].head(k)
