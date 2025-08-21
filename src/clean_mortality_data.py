import pandas as pd

# Load raw mortality data
file_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/opioid_mortality_raw.csv"
df_raw = pd.read_csv(file_path)

# Drop extra 'Notes' column if exists
if 'Notes' in df_raw.columns:
    df_raw = df_raw.drop(columns=['Notes'])

# Preview actual column names
print("📋 Columns in the raw file:")
print(df_raw.columns.tolist())

# Select and rename relevant columns
df = df_raw[['Year', 'State', 'Deaths', 'Population']].copy()
df.columns = ['year', 'state', 'deaths', 'population']

# Clean death counts
df = df[df['deaths'].apply(lambda x: str(x).replace(',', '').replace('.', '').isdigit())]
df['deaths'] = df['deaths'].apply(lambda x: int(float(str(x).replace(',', ''))))

# Save cleaned version
output_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/opioid_mortality_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"✅ Cleaned data saved to: {output_path}")
