# scripts/als_smoke.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data_utils import load_ratings
from src.models.als_implicit import ALSImplicitRecommender, ALSParams

CSV = Path("data/updated_ratings_with_reviews.csv")

def make_dense_subset(df: pd.DataFrame, n_users=5000, n_items=5000,
                      min_per_user=3, min_per_item=3):
    top_users = df["userId"].value_counts().head(n_users).index
    top_items = df["productId"].value_counts().head(n_items).index
    sub = df[df["userId"].isin(top_users) & df["productId"].isin(top_items)]
    sub = sub.groupby("userId").filter(lambda x: len(x) >= min_per_user)
    sub = sub.groupby("productId").filter(lambda x: len(x) >= min_per_item)
    return sub

def main():
    df = load_ratings(str(CSV))
    sub = make_dense_subset(df, n_users=4000, n_items=4000)
    u = int(sub["userId"].value_counts().index[0])

    params = ALSParams(factors=64, regularization=0.02, iterations=15, alpha=40.0)
    model = ALSImplicitRecommender(params).fit(sub)
    recs = model.recommend_for_user(u, top_k=5)
    print("User:", u)
    print(recs)

if __name__ == "__main__":
    main()