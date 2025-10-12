# scripts/svd_smoke.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data_utils import load_ratings
from src.models.svd_surprise import SVDSurpriseRecommender as SVD

CSV = Path("data/updated_ratings_with_reviews.csv")

def make_dense_subset(df: pd.DataFrame, n_users=5000, n_items=5000,
                      min_per_user=3, min_per_item=3):
    # SVD is lighter than KNN, so we can use a larger subset, but keep it reasonable
    top_users = df["userId"].value_counts().head(n_users).index
    top_items = df["productId"].value_counts().head(n_items).index
    sub = df[df["userId"].isin(top_users) & df["productId"].isin(top_items)]
    sub = sub.groupby("userId").filter(lambda x: len(x) >= min_per_user)
    sub = sub.groupby("productId").filter(lambda x: len(x) >= min_per_item)
    return sub

def main():
    df = load_ratings(str(CSV))
    sub = make_dense_subset(df, n_users=4000, n_items=4000)  # tweak if slow
    print(f"Subset: {sub.shape}, users={sub.userId.nunique()}, items={sub.productId.nunique()}")

    # pick a reasonable user (not too sparse)
    vc = sub["userId"].value_counts()
    cand_users = vc[(vc >= 3) & (vc <= 80)].index
    if len(cand_users) == 0:
        cand_users = vc.index
    test_user = int(cand_users[0])

    model = SVD(n_factors=80, n_epochs=15).fit(sub)
    recs = model.recommend_for_user(test_user, top_k=5)
    print("Test user:", test_user)
    print(recs)

if __name__ == "__main__":
    main()
