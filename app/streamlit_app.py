# app/streamlit_app.py
import sys
from pathlib import Path

# make Python see project root so we can import src/*
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.data_utils import load_ratings
from src.models.popularity import PopularityRecommender

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "updated_ratings_with_reviews.csv"

st.set_page_config(page_title="E-Commerce Recommender", page_icon="🛒")
st.title("🛒 Intelligent E-Commerce Recommendation System")
st.caption("Step 2: Popularity baseline")

# ---- load data ----
try:
    df = load_ratings(str(DATA_PATH))
    st.success(f"✅ Loaded {len(df):,} rows from {DATA_PATH.name}")
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

with st.expander("Preview data (first 10 rows)", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# ---- popularity controls ----
st.subheader("Popularity Recommender")
min_r = st.slider("Minimum ratings per product", 1, 50, 5)
k = st.slider("How many recommendations?", 1, 50, 10)

# ---- fit + show ----
model = PopularityRecommender(min_ratings=min_r).fit(df)
topk = model.top_k(k)

st.write("**Top items by popularity score**")
st.dataframe(topk, use_container_width=True)
st.caption("Score = average rating × sqrt(number of ratings).")
