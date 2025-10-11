import pandas as pd

# updated to match your actual dataset
REQUIRED = ["mapped_user_id", "mapped_product_id", "rating"]

def load_ratings(path: str) -> pd.DataFrame:
    """
    Loads the ratings CSV and ensures columns and types are correct.
    """
    df = pd.read_csv(path)

    # check for required columns
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # rename to consistent names (for models)
    df = df.rename(columns={
        "mapped_user_id": "userId",
        "mapped_product_id": "productId",
        "rating": "Rating"
    })

    # make sure correct types
    df["userId"] = df["userId"].astype(int)
    df["productId"] = df["productId"].astype(int)
    df["Rating"] = df["Rating"].astype(float)

    return df
