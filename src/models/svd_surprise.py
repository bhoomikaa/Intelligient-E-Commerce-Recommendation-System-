# src/models/svd_surprise.py
import pandas as pd
from surprise import Dataset, Reader, SVD

class SVDSurpriseRecommender:
    """
    Matrix Factorization (SVD) using Surprise.
    Expects df with columns: userId, productId, Rating.
    We cast IDs to strings to keep Surprise's raw-id mapping consistent.
    """

    def __init__(self, n_factors: int = 100, n_epochs: int = 20, lr_all: float = 0.005, reg_all: float = 0.02):
        self.params = dict(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        self.algo = None
        self.trainset = None
        self.all_items_inner = None

    def fit(self, ratings: pd.DataFrame):
        df2 = ratings.copy()
        df2["userId"] = df2["userId"].astype(str)
        df2["productId"] = df2["productId"].astype(str)

        reader = Reader(rating_scale=(float(df2["Rating"].min()), float(df2["Rating"].max())))
        data = Dataset.load_from_df(df2[["userId", "productId", "Rating"]], reader)
        trainset = data.build_full_trainset()

        algo = SVD(**self.params)
        algo.fit(trainset)

        self.algo = algo
        self.trainset = trainset
        self.all_items_inner = set(trainset.all_items())
        return self

    def recommend_for_user(self, raw_user_id: int | str, top_k: int = 10) -> pd.DataFrame:
        import pandas as pd
        if self.algo is None or self.trainset is None:
            raise RuntimeError("Call fit() first.")

        raw_uid = str(raw_user_id)
        # user must exist
        if raw_uid not in self.trainset._raw2inner_id_users:
            return pd.DataFrame(columns=["productId", "est"])

        inner_uid = self.trainset.to_inner_uid(raw_uid)

        rated_items = {inner_iid for (inner_iid, _) in self.trainset.ur[inner_uid]}
        candidates = self.all_items_inner - rated_items
        if not candidates:
            return pd.DataFrame(columns=["productId", "est"])

        preds = []
        for inner_iid in candidates:
            raw_iid = self.trainset.to_raw_iid(inner_iid)  # string
            est = self.algo.predict(raw_uid, raw_iid).est
            preds.append((int(raw_iid), float(est)))

        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:top_k]
        return pd.DataFrame(top, columns=["productId", "est"])
    
    