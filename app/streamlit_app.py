import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "updated_ratings_with_reviews.csv"

st.set_page_config(page_title="E-Commerce Recommender", page_icon="🛒")
st.title("🛒 Intelligent E-Commerce Recommendation System")
st.caption("Step 1: Data preview — verifying setup")

try:
    df = pd.read_csv(DATA_PATH)
    st.success(f"✅ Loaded {len(df):,} rows from {DATA_PATH.name}")
    st.subheader("Sample of the dataset:")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Columns:**", ", ".join(df.columns))
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
