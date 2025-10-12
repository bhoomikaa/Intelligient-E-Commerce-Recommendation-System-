# src/models/als_implicit.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import implicit  # pip/conda already installed

@dataclass
class ALSParams:
    factors: int = 64
    regularization: float = 0.01
    iterations: int = 15
    alpha: float = 40.0  # confidence scaling for implicit feedback

class ALSImplicitRecommender:
    """
    Train an implicit ALS model on user-item interactions.
    We treat Rating as implicit strength (confidence).
    Expects columns: userId, productId, Rating
    """

    def __init__(self, params: ALSParams | None = None):
        self.p = params or ALSParams()
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.p.factors,
            regularization=self.p.regularization,
            iterations=self.p.iterations,
            calculate_training_loss=False
        )
        self.user2inner = None
        self.item2inner = None
        self.inner2user = None
        self.inner2item = None
        self.user_items_csr: csr_matrix | None = None
        self.item_users_csr: csr_matrix | None = None

    def _build_mappings(self, df: pd.DataFrame):
        # make consecutive ids for matrix rows/cols
        users, user_index = np.unique(df["userId"].values, return_inverse=True)
        items, item_index = np.unique(df["productId"].values, return_inverse=True)
        self.user2inner = {int(u): i for i, u in enumerate(users)}
        self.item2inner = {int(i_): j for j, i_ in enumerate(items)}
        self.inner2user = {i: int(u) for i, u in enumerate(users)}
        self.inner2item = {j: int(i_) for j, i_ in enumerate(items)}
        return user_index, item_index, len(users), len(items)

    def fit(self, ratings: pd.DataFrame):
        # keep only the needed columns
        df = ratings.loc[:, ["userId", "productId", "Rating"]].copy()

        # optional: clip/scale ratings for stability as implicit confidence
        df["Rating"] = df["Rating"].astype(float).clip(lower=0.0)

        # maps to consecutive ids
        u_idx, i_idx, n_users, n_items = self._build_mappings(df)

        # confidence-weighted interaction matrix (users x items)
        # ALS from 'implicit' expects item-user CSR for training
        data = df["Rating"].values.astype(np.float32) * self.p.alpha
        mat_ui = coo_matrix((data, (u_idx, i_idx)), shape=(n_users, n_items)).tocsr()
        mat_iu = mat_ui.T.tocsr()

        self.user_items_csr = mat_ui
        self.item_users_csr = mat_iu

        # fit ALS (on item-user)
        self.model.fit(self.item_users_csr)
        return self

    def recommend_for_user(self, raw_user_id: int, top_k: int = 10):
        import pandas as pd
        if self.user_items_csr is None:
            raise RuntimeError("Call fit() first.")

        uid = int(raw_user_id)
        if uid not in self.user2inner:
            return pd.DataFrame(columns=["productId", "score"])

        u_inner = self.user2inner[uid]

    # Pass ONLY the row for this user (shape = 1 x n_items)
    # Newer implicit versions require user_items.shape[0] == len(userids)
        user_row = self.user_items_csr[u_inner]

        rec_items, rec_scores = self.model.recommend(
            userid=u_inner,
            user_items=user_row,
            N=top_k,
            filter_already_liked_items=True,
            recalculate_user=True,  # compute user factors from this row if needed
        )

        rows = [
            {"productId": self.inner2item[int(inner_i)], "score": float(s)}
            for inner_i, s in zip(rec_items, rec_scores)
        ]
        return pd.DataFrame(rows, columns=["productId", "score"])