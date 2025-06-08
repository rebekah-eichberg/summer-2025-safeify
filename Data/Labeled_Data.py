import pandas as pd

# Step 1: Load both files
df_meta = pd.read_json('Data/amazon_meta.json', lines=True)
df_labels = pd.read_csv('Data/amazon_df_labels.csv')

# Step 2: Check row count
if len(df_meta) != len(df_labels):
    raise ValueError("❌ Row count mismatch — cannot merge by index.")

# Step 3: Check if ASINs match in order
asin_match = df_meta['asin'].reset_index(drop=True).equals(df_labels['asin'].reset_index(drop=True))
if not asin_match:
    raise ValueError("❌ ASINs do not match in order — cannot merge by index.")

# Step 4: Drop 'asin' from labels to avoid duplication
df_labels_cleaned = df_labels.drop(columns=['asin'])

# Step 5: Merge naively by index
df_merged = pd.concat([df_meta.reset_index(drop=True), df_labels_cleaned.reset_index(drop=True)], axis=1)

# Step 6: Save
df_merged.to_csv('Data/amazon_merged.csv', index=False)
print("✅ Saved: Data/amazon_merged.csv")
