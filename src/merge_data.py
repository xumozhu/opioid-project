import pandas as pd

# Step 1: Load cleaned datasets
mortality_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/opioid_mortality_cleaned.csv"
policy_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/pdmp_cleaned.csv"

df_mortality = pd.read_csv(mortality_path)
df_policy = pd.read_csv(policy_path)

# Step 2: Standardize state names to lowercase for merging
df_mortality['state'] = df_mortality['state'].str.strip().str.lower()
df_policy['state'] = df_policy['state'].str.strip().str.lower()

# Step 3: Merge on state and year
df_merged = pd.merge(df_mortality, df_policy, on=['state', 'year'], how='inner')

# Step 4: Preview merged data
print("📊 Merged dataset preview:")
print(df_merged.head())

# Step 5: Save final dataset
output_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data.csv"
df_merged.to_csv(output_path, index=False)
print(f"✅ Final merged data saved to: {output_path}")
