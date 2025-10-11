# app/streamlit_app.py
import sys
from pathlib import Path

# 1) make Python see the project root so "src" can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.data_utils import load_ratings

# 2) path to your CSV
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "updated_ratings_with_reviews.csv"

# 3) basic page setup
st.set_page_config(page_title="E-Commerce Recommender", page_icon="🛒")
st.title("🛒 Intelligent E-Commerce Recommendation System")
st.caption("Step 1: Data preview — now using our loader")

# 4) load data with our helper
try:
    df = load_ratings(str(DATA_PATH))
    st.success(f"✅ Loaded {len(df):,} rows from {DATA_PATH.name}")
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

# 5) preview
with st.expander("Preview data (first 10 rows)", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

st.write("**Columns after standardizing:**", ", ".join(df.columns))
st.caption("We renamed mapped_user_id→userId, mapped_product_id→productId, rating→Rating for consistency.")
