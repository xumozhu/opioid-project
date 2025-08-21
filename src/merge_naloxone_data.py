import pandas as pd


main_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_withpoverty.csv"
naloxone_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/naloxone_cleaned.csv"
output_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv"


main_df = pd.read_csv(main_path)
naloxone_df = pd.read_csv(naloxone_path)

print("Main DF columns:", main_df.columns.tolist())
print("Naloxone DF columns:", naloxone_df.columns.tolist())

merged_df = pd.merge(main_df, naloxone_df, on=["state", "year"], how="left")

print("✅ Merged shape:", merged_df.shape)
print("✅ New columns:", merged_df.columns.tolist())

merged_df.to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")
