import pandas as pd
from pathlib import Path

input_path = Path("/Users/zhuxumo/Desktop/opioid-project/datasets/naloxone_raw.csv")
output_path = Path("/Users/zhuxumo/Desktop/opioid-project/datasets/naloxone_cleaned.csv")

df_raw = pd.read_csv(input_path)

df = df_raw[[
    'Jurisdictions',
    'Effective Date',
    'pharmacist-dispensing',
    'pharmacist-dispensing-method_Standing order',
    'pharmacist-dispensing-method_Protocol order',
    'pharmacist-dispensing-method_Naloxone-specific collaborative practice agreement',
    'pharmacist-dispensing-method_Pharmacist prescriptive authority',
    'pharmacist-dispensing-method_Directly authorized by legislature'
]].copy()

df.rename(columns={"Jurisdictions": "state"}, inplace=True)

df["year"] = pd.to_datetime(df["Effective Date"], errors='coerce').dt.year

df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

dispense_cols = [col for col in df.columns if col.startswith("pharmacist-dispensing-method")]
df[dispense_cols] = df[dispense_cols].applymap(lambda x: 1 if str(x).strip() == "1" else 0)

df["naloxone_access"] = df[dispense_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

df = df[df["naloxone_access"] == 1]

df_final = df[["state", "year", "naloxone_access"]].drop_duplicates()

all_states = df_raw['Jurisdictions'].unique()

years = list(range(2010, 2023))  

full_grid = pd.MultiIndex.from_product([all_states, years], names=['state', 'year']).to_frame(index=False)

df_merged = pd.merge(full_grid, df_final, on=['state', 'year'], how='left')
df_merged['naloxone_access'] = df_merged['naloxone_access'].fillna(0).astype(int)

df_final = df_merged.sort_values(['state', 'year'])


df_final.to_csv(output_path, index=False)
print("✅ naloxone_cleaned.csv saved successfully!")
