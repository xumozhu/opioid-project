import pandas as pd
import os
import glob

folder_path = "/Users/zhuxumo/Desktop/opioid-project/datasets"
file_pattern = os.path.join(folder_path, "poverty_*.csv")
files = sorted(glob.glob(file_pattern))
files = [f for f in files if "cleaned" not in f]

cleaned_data = []

valid_states = set([
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", 
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", 
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", 
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", 
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", 
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
    "Wisconsin", "Wyoming"
])

for file in files:
    year = int(os.path.basename(file).split("_")[1].split(".")[0])
    
    df = pd.read_csv(file, skiprows=2)
    
    df = df[df["Location"].isin(valid_states)]

    df_cleaned = df[["Location", "Total"]].copy()
    df_cleaned.columns = ["state", "poverty_population"]
    df_cleaned["year"] = year
    
    cleaned_data.append(df_cleaned)

final_df = pd.concat(cleaned_data, ignore_index=True)

final_df.to_csv("/Users/zhuxumo/Desktop/opioid-project/datasets/poverty_cleaned.csv", index=False)
print(final_df.head())
