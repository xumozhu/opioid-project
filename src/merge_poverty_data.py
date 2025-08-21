import pandas as pd

main_df = pd.read_csv("/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data.csv")
poverty_df = pd.read_csv("/Users/zhuxumo/Desktop/opioid-project/datasets/poverty_cleaned.csv")

main_df['state'] = main_df['state'].str.strip().str.title()
poverty_df['state'] = poverty_df['state'].str.strip().str.title()

main_df['year'] = main_df['year'].astype(int)
poverty_df['year'] = poverty_df['year'].astype(int)

merged_df = pd.merge(main_df, poverty_df, on=['state', 'year'], how='left')

unmatched = merged_df[merged_df['poverty_population'].isna()][['state', 'year']].drop_duplicates()
print("⚠️ can't find state-year in poverty_cleaned.csv:")
print(unmatched)

merged_df.to_csv("/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_withpoverty.csv", index=False)
print("✅ success, merged_data_withpoverty")
