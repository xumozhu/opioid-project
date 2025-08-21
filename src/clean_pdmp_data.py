import pandas as pd

# Step 1: Load the cleaned PDMP source file
file_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/pdmp_raw.csv"
df_raw = pd.read_csv(file_path)

# Step 2: Extract relevant columns
df = df_raw[['Jurisdictions', 'pdmpimp-op']].copy()
df.columns = ['state', 'operational_date']

# Step 3: Convert operational date to year
df['year_implemented'] = pd.to_datetime(df['operational_date'], errors='coerce').dt.year
df = df.dropna(subset=['year_implemented'])

# Step 4: Build state-year panel from 2010–2022
states = df['state'].unique()
years = list(range(2010, 2023))

records = []
for state in states:
    year_impl = int(df[df['state'] == state]['year_implemented'].values[0])
    for year in years:
        records.append({
            'state': state,
            'year': year,
            'pdmp_implemented': 1 if year >= year_impl else 0
        })

df_panel = pd.DataFrame(records)

# Step 5: Save panel-format data
output_path = "/Users/zhuxumo/Desktop/opioid-project/datasets/pdmp_cleaned.csv"
df_panel.to_csv(output_path, index=False)
print(f"✅ Cleaned PDMP data saved to: {output_path}")
