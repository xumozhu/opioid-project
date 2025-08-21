import os
import pandas as pd

input_folder = '/Users/zhuxumo/Desktop/opioid-project/Unemployment rate'     
output_folder = '/Users/zhuxumo/Desktop/opioid-project/Unemployment_rate_csv'      

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.xlsx'):
        xlsx_path = os.path.join(input_folder, filename)
        csv_filename = filename.replace('.xlsx', '.csv')
        csv_path = os.path.join(output_folder, csv_filename)

        df = pd.read_excel(xlsx_path)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"✅ transferred: {filename} → {csv_filename}")
