# scripts/knn_smoke.py
from pathlib import Path
import pandas as pd

from src.data_utils import load_ratings
from src.models.knn_surprise import KNNSurpriseRecommender as KNN

CSV = Path("data/updated_ratings_with_reviews.csv")

def make_dense_subset(df: pd.DataFrame, n_users=2000, n_items=2000, min_per_user=3, min_per_item=3):
    """Keep the most active users/items to avoid huge similarity matrices."""
    top_users = df["userId"].value_counts().head(n_users).index
    top_items = df["productId"].value_counts().head(n_items).index
    sub = df[df["userId"].isin(top_users) & df["productId"].isin(top_items)]

    # ensure a little density
    sub = sub.groupby("userId").filter(lambda x: len(x) >= min_per_user)
    sub = sub.groupby("productId").filter(lambda x: len(x) >= min_per_item)
    return sub

def main():
    df = load_ratings(str(CSV))
    sub = make_dense_subset(df, n_users=2000, n_items=2000)
    print(f"Subset shape: {sub.shape}, users={sub.userId.nunique()}, items={sub.productId.nunique()}")

    # Choose a user with a moderate number of ratings (so they have unseen items)
    vc = sub["userId"].value_counts()
    cand_users = vc[(vc >= 3) & (vc <= 50)].index
    if len(cand_users) == 0:
        cand_users = vc.index  # fall back to any user
    test_user = int(cand_users[0])

    rated = sub[sub.userId == test_user]["productId"].nunique()
    total_items = sub["productId"].nunique()
    print(f"Picked user {test_user} who rated {rated} / {total_items} items")

    # Item-based (smaller sim matrix). You can change to user_based=True later.
    model = KNN(sim_name="cosine", user_based=False, k=40).fit(sub)

    recs = model.recommend_for_user(test_user, top_k=5)
    print(recs)

if __name__ == "__main__":
    main()
