import pandas as pd

main_data_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_with_unemployment.csv"
income_data_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/median_income_cleaned.csv"
output_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv"

main_df = pd.read_csv(main_data_path)
income_df = pd.read_csv(income_data_path)

main_df['state'] = main_df['state'].str.upper()
income_df['state'] = income_df['state'].str.upper()

print("🔍 ：", income_df.columns)

income_column = [col for col in income_df.columns if 'income' in col.lower()]
if income_column:
    income_column = income_column[0]
else:
    raise ValueError("❗️can't find'income'")

income_df = income_df.rename(columns={income_column: 'median_household_income'})

merged_df = pd.merge(main_df, income_df, on=['state', 'year'], how='left')

missing = merged_df['median_household_income'].isnull().sum()
print(f"❗️Missing median_income rows after merge: {missing}")

merged_df.to_csv(output_path, index=False)
print("✅ success merged_data_final.csv")
