import pandas as pd

main_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_this.csv"
medicaid_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/medicaid_expansion_cleaned.csv"
output_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv"

main_df = pd.read_csv(main_path)
medicaid_df = pd.read_csv(medicaid_path)

merged_df = pd.merge(main_df, medicaid_df, how='left', on=['state', 'year'])

missing = merged_df['medicaid_expansion'].isna().sum()
if missing > 0:
    print(f"⚠️ {missing}")
   
    merged_df['medicaid_expansion'] = merged_df['medicaid_expansion'].fillna(0)

merged_df.to_csv(output_path, index=False)
print(f"✅ success：{output_path}")
