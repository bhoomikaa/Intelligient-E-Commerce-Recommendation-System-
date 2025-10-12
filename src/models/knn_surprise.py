# src/models/knn_surprise.py
from typing import List
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans

class KNNSurpriseRecommender:
    """
    User-based or item-based KNN using Surprise.
    Expects a DataFrame with columns: userId, productId, Rating.
    We cast IDs to strings to keep Surprise's raw-id mapping consistent.
    """

    def __init__(self, sim_name: str = "cosine", user_based: bool = True, k: int = 50, min_k: int = 1):
        self.sim_name = sim_name
        self.user_based = user_based
        self.k = k
        self.min_k = min_k
        self.algo = None
        self.trainset = None
        self.all_items_inner = None  # set of all inner item ids

    def fit(self, ratings: pd.DataFrame):
        # ---- IMPORTANT: cast IDs to str for Surprise raw-ID mapping
        df2 = ratings.copy()
        df2["userId"] = df2["userId"].astype(str)
        df2["productId"] = df2["productId"].astype(str)

        reader = Reader(rating_scale=(float(df2["Rating"].min()), float(df2["Rating"].max())))
        data = Dataset.load_from_df(df2[["userId", "productId", "Rating"]], reader)
        trainset = data.build_full_trainset()

        sim_options = {"name": self.sim_name, "user_based": self.user_based}
        algo = KNNWithMeans(k=self.k, min_k=self.min_k, sim_options=sim_options)
        algo.fit(trainset)

        self.algo = algo
        self.trainset = trainset
        self.all_items_inner = set(trainset.all_items())
        return self

    def recommend_for_user(self, raw_user_id: int | str, top_k: int = 10) -> pd.DataFrame:
        """
        Predict ratings for items the user hasn't rated yet and return top_k.
        Returns empty DataFrame if user isn't in the trainset or has no candidates.
        """
        import pandas as pd

        if self.algo is None or self.trainset is None:
            raise RuntimeError("Call fit() first.")

        # Use string raw IDs consistently
        raw_uid = str(raw_user_id)

        # user must exist
        if raw_uid not in self.trainset._raw2inner_id_users:
            return pd.DataFrame(columns=["productId", "est"])

        inner_uid = self.trainset.to_inner_uid(raw_uid)

        # items already rated by this user (inner ids)
        rated_items = {inner_iid for (inner_iid, _) in self.trainset.ur[inner_uid]}

        # candidate items = all items - rated
        candidates = self.all_items_inner - rated_items
        if not candidates:
            return pd.DataFrame(columns=["productId", "est"])

        preds = []
        for inner_iid in candidates:
            raw_iid = self.trainset.to_raw_iid(inner_iid)  # already a string
            est = self.algo.predict(raw_uid, raw_iid).est
            preds.append((int(raw_iid), float(est)))  # cast back to int for display

        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:top_k]
        return pd.DataFrame(top, columns=["productId", "est"])
