import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 路径
DATA_PATH = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv"
OUTPUT_PATH = "/Users/zhuxumo/Desktop/opioid-project/outputs"

# 读取数据
data = pd.read_csv(DATA_PATH)

# ---------- 图3：Poverty vs Deaths ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='poverty_population', y='deaths', hue='year', palette='viridis')
plt.title("Scatterplot of Poverty Population vs Opioid Deaths")
plt.xlabel("Poverty Population")
plt.ylabel("Opioid Deaths")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "poverty_vs_deaths.png"))
plt.close()

print("✅ Poverty vs. Deaths plot saved.")
