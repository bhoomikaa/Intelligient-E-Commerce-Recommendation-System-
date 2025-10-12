# app/streamlit_app.py
import sys
from pathlib import Path

# Let Python see the project root so we can import from src/*
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd

from src.data_utils import load_ratings
from src.models.popularity import PopularityRecommender
from src.models.knn_surprise import KNNSurpriseRecommender
from src.models.svd_surprise import SVDSurpriseRecommender
from src.models.als_implicit import ALSImplicitRecommender, ALSParams

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "updated_ratings_with_reviews.csv"

st.set_page_config(page_title="E-Commerce Recommender", page_icon="🛒", layout="centered")
st.title("🛒 Intelligent E-Commerce Recommendation System")

# ---------- utilities ----------
@st.cache_data(show_spinner=False)
def load_df(csv_path: str) -> pd.DataFrame:
    return load_ratings(csv_path)

def make_dense_subset(df: pd.DataFrame, n_users=1200, n_items=1200,
                      min_per_user=3, min_per_item=3):
    top_users = df["userId"].value_counts().head(n_users).index
    top_items = df["productId"].value_counts().head(n_items).index
    sub = df[df["userId"].isin(top_users) & df["productId"].isin(top_items)]
    sub = sub.groupby("userId").filter(lambda x: len(x) >= min_per_user)
    sub = sub.groupby("productId").filter(lambda x: len(x) >= min_per_item)
    return sub

# ---------- data ----------
try:
    df = load_df(str(DATA_PATH))
    st.success(f"✅ Loaded {len(df):,} rows from {DATA_PATH.name}")
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

with st.expander("Preview data (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# ---------- sidebar model selector ----------
st.sidebar.header("Model")
model_choice = st.sidebar.selectbox(
    "Choose a recommender",
    ["Popularity", "KNN (Surprise)", "SVD (Surprise)", "ALS (implicit)"],
    index=0,
)

# =========================================================
# POPULARITY
# =========================================================
def render_popularity(df: pd.DataFrame):
    st.subheader("Popularity Recommender")
    min_r = st.slider("Minimum ratings per product", 1, 50, 5, key="pop_min_r")
    k = st.slider("How many recommendations?", 1, 50, 10, key="pop_k")
    model = PopularityRecommender(min_ratings=min_r).fit(df)
    topk = model.top_k(k)
    st.write("**Top items by popularity score**")
    st.dataframe(topk, use_container_width=True)
    st.caption("Score = average rating × sqrt(number of ratings).")

# =========================================================
# KNN (Surprise)
# =========================================================
def render_knn(df: pd.DataFrame):
    st.subheader("KNN Collaborative Filtering (Surprise)")
    colA, colB, colC = st.columns(3)
    with colA:
        mode = st.radio("Mode", ["Item-based", "User-based"], index=0, key="knn_mode")
    with colB:
        sim_name = st.selectbox("Similarity", ["cosine", "msd", "pearson"], index=0, key="knn_sim")
    with colC:
        rec_k = st.slider("Recommendations", 1, 20, 5, key="knn_rec_k")

    col1, col2 = st.columns(2)
    with col1:
        n_users_knn = st.slider("Subset: #users", 500, 3000, 1200, step=100, key="knn_users")
    with col2:
        n_items_knn = st.slider("Subset: #items", 500, 3000, 1200, step=100, key="knn_items")

    sample_users = df["userId"].value_counts().index.tolist()[:2000]
    default_user = int(sample_users[0]) if sample_users else 0
    user_id_knn = st.number_input("Enter userId", min_value=0, value=default_user, step=1, key="knn_user_input")

    if st.button("Recommend with KNN"):
        with st.spinner("Training KNN on a dense subset…"):
            sub = make_dense_subset(df, n_users=n_users_knn, n_items=n_items_knn)
            if user_id_knn not in sub["userId"].values:
                st.warning("That userId is not in the current subset. Increase subset size or pick another user.")
                return
            try:
                model_knn = KNNSurpriseRecommender(
                    sim_name=sim_name,
                    user_based=(mode == "User-based"),
                    k=40,
                    min_k=1,
                ).fit(sub)
                recs = model_knn.recommend_for_user(user_id_knn, top_k=rec_k)
                if recs.empty:
                    st.info("No unseen items to recommend within this subset.")
                else:
                    st.dataframe(recs, use_container_width=True)
            except Exception as e:
                st.error(f"KNN failed: {e}")

# =========================================================
# SVD (Surprise)
# =========================================================
def render_svd(df: pd.DataFrame):
    st.subheader("SVD (Surprise) — Matrix Factorization")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_factors = st.slider("n_factors", 20, 200, 80, step=10, key="svd_nf")
    with c2:
        n_epochs = st.slider("n_epochs", 5, 50, 15, step=5, key="svd_ne")
    with c3:
        rec_k_svd = st.slider("Recommendations", 1, 20, 5, key="svd_k")
    with c4:
        sub_scale = st.selectbox(
            "Subset size",
            ["Small (1k/1k)", "Medium (2k/2k)", "Large (4k/4k)"],
            index=1,
            key="svd_subset",
        )
    preset = {
        "Small (1k/1k)": (1000, 1000),
        "Medium (2k/2k)": (2000, 2000),
        "Large (4k/4k)": (4000, 4000),
    }
    n_users_svd, n_items_svd = preset[sub_scale]

    sample_users = df["userId"].value_counts().index.tolist()[:2000]
    default_user = int(sample_users[0]) if sample_users else 0
    user_id_svd = st.number_input("Enter userId", min_value=0, value=default_user, step=1, key="svd_user_input")

    if st.button("Recommend with SVD"):
        with st.spinner("Training SVD on subset…"):
            sub = make_dense_subset(df, n_users=n_users_svd, n_items=n_items_svd)
            if user_id_svd not in sub["userId"].values:
                st.warning("That userId is not in the subset. Choose a larger subset or a different user.")
                return
            try:
                model_svd = SVDSurpriseRecommender(n_factors=n_factors, n_epochs=n_epochs).fit(sub)
                recs = model_svd.recommend_for_user(user_id_svd, top_k=rec_k_svd)
                if recs.empty:
                    st.info("No unseen items to recommend within this subset.")
                else:
                    st.dataframe(recs, use_container_width=True)
            except Exception as e:
                st.error(f"SVD failed: {e}")

# =========================================================
# ALS (implicit)
# =========================================================
def render_als(df: pd.DataFrame):
    st.subheader("ALS (implicit) — Matrix Factorization on implicit feedback")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        als_factors = st.slider("factors", 16, 256, 64, step=16, key="als_factors")
    with c2:
        als_reg = st.selectbox("regularization", [0.002, 0.005, 0.01, 0.02, 0.05], index=3, key="als_reg")
    with c3:
        als_iters = st.slider("iterations", 5, 50, 15, step=5, key="als_iters")
    with c4:
        als_alpha = st.selectbox("alpha (confidence)", [10.0, 20.0, 40.0, 80.0], index=2, key="als_alpha")

    c5, c6 = st.columns(2)
    with c5:
        n_users_als = st.slider("Subset: #users", 1000, 10000, 4000, step=500, key="als_users")
    with c6:
        n_items_als = st.slider("Subset: #items", 1000, 10000, 4000, step=500, key="als_items")

    sample_users = df["userId"].value_counts().index.tolist()[:2000]
    default_user = int(sample_users[0]) if sample_users else 0
    user_id_als = st.number_input("Enter userId", min_value=0, value=default_user, step=1, key="als_user")

    if st.button("Recommend with ALS"):
        with st.spinner("Training ALS on subset…"):
            sub = make_dense_subset(df, n_users=n_users_als, n_items=n_items_als)
            if user_id_als not in sub["userId"].values:
                st.warning("That userId is not in the subset. Increase subset size or pick another user.")
                return
            try:
                params = ALSParams(
                    factors=als_factors,
                    regularization=float(als_reg),
                    iterations=als_iters,
                    alpha=float(als_alpha),
                )
                als_model = ALSImplicitRecommender(params).fit(sub)
                recs = als_model.recommend_for_user(user_id_als, top_k=5)
                if recs.empty:
                    st.info("No unseen items to recommend within this subset.")
                else:
                    st.dataframe(recs, use_container_width=True)
            except Exception as e:
                st.error(f"ALS failed: {e}")

# ---------- route to selected model ----------
if model_choice == "Popularity":
    render_popularity(df)
elif model_choice == "KNN (Surprise)":
    render_knn(df)
elif model_choice == "SVD (Surprise)":
    render_svd(df)
else:
    render_als(df)