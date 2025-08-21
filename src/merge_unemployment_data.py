import pandas as pd

# File paths
main_data_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_this.csv"
unemployment_data_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/unemploymentRate_cleaned.csv"
output_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_with_unemployment.csv"

# Load datasets
main_df = pd.read_csv(main_data_path)
unemp_df = pd.read_csv(unemployment_data_path)

# Standardize state names if necessary
main_df['state'] = main_df['state'].str.upper()
unemp_df['state'] = unemp_df['state'].str.upper()

# Merge on year and state
merged_df = pd.merge(main_df, unemp_df, on=['year', 'state'], how='left')

# Check for missing values
missing = merged_df['unemployment_rate'].isnull().sum()
print(f"✅ Merge completed. Missing unemployment_rate values: {missing}")

# Save merged file
merged_df.to_csv(output_path, index=False)
print(f"✅ New dataset saved to: {output_path}")
