import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv"
OUTPUT_PATH = "/Users/zhuxumo/Desktop/opioid-project/outputs"

data = pd.read_csv(DATA_PATH)

pdmp_by_year = data.groupby('year')['pdmp_implemented'].mean()

plt.figure(figsize=(10, 6))
plt.plot(pdmp_by_year.index, pdmp_by_year.values, marker='o', color='royalblue')
plt.title("Proportion of States with PDMP Implemented Over Time")
plt.xlabel("Year")
plt.ylabel("Proportion of States")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "pdmp_adoption_trend.png"))
plt.close()

naloxone_by_year = data.groupby('year')['naloxone_access'].mean()

plt.figure(figsize=(10, 6))
plt.plot(naloxone_by_year.index, naloxone_by_year.values, marker='o', color='darkorange')
plt.title("Proportion of States with Naloxone Access Law Over Time")
plt.xlabel("Year")
plt.ylabel("Proportion of States")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "naloxone_adoption_trend.png"))
plt.close()

print("✅ PDMP & Naloxone policy adoption trend plots saved.")
