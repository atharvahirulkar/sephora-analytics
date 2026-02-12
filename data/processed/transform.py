import pandas as pd
import glob
import os

DATA_DIR = "/Users/atharva/ATHARVA/UCSD/Winter26/202_DtMgmt/Project/seph"

# -------------------------------------------------------
# 1. Load all review chunks
# -------------------------------------------------------
review_files = glob.glob(os.path.join(DATA_DIR, "reviews_*.csv"))
print(f"Found {len(review_files)} review files")

assert len(review_files) > 0, "No review CSV files found!"

dfs = []
for f in review_files:
    print("Loading:", f)
    dfs.append(pd.read_csv(f, low_memory=False))

reviews = pd.concat(dfs, ignore_index=True)

# -------------------------------------------------------
# 2. Clean
# -------------------------------------------------------
# Drop useless index column
reviews = reviews.drop(columns=["Unnamed: 0"], errors="ignore")

# Ensure text is string
reviews["review_text"] = reviews["review_text"].astype(str)
reviews["review_title"] = reviews["review_title"].astype(str)

# -------------------------------------------------------
# 3. Dataset sanity checks
# -------------------------------------------------------
print("\n=== Sephora Dataset Summary ===")
print("Total rows:", len(reviews))
print("Unique users:", reviews["author_id"].nunique())
print("Unique products:", reviews["product_id"].nunique())
print("Unique brands:", reviews["brand_name"].nunique())
print("Avg review length:", reviews["review_text"].str.len().mean())
print("Missing review text:", reviews["review_text"].isna().mean())
print("Missing ratings:", reviews["rating"].isna().mean())

# -------------------------------------------------------
# 4. Save unified analytic dataset
# -------------------------------------------------------
out_path = os.path.join(DATA_DIR, "sephora_full.csv")
reviews.to_csv(out_path, index=False)

print("\nSaved final dataset to:", out_path)
print("================================")
