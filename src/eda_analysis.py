import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

data = pd.read_csv("/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv")

data['death_rate_per_100k'] = data['deaths'] / data['population'] * 100000

# ========== 1. Overall Trend of Total Deaths ==========
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='year', y='deaths', estimator='sum', ci=None)
plt.title("Total Opioid Overdose Deaths in the U.S. Over Time")
plt.ylabel("Total Deaths")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/overall_death_trend.png")
plt.close()

# ========== 2. Annual Average Death Rate Trend ==========
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='year', y='death_rate_per_100k', estimator='mean', ci=None)
plt.title("Average Opioid Death Rate per 100k Over Time")
plt.ylabel("Death Rate per 100k")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/avg_death_rate_trend.png")
plt.close()

# ========== 3. Comparison of PDMP Implementation ==========
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='pdmp_implemented', y='death_rate_per_100k', estimator='mean')
plt.title("Average Death Rate by PDMP Implementation")
plt.xticks([0, 1], ["Not Implemented", "Implemented"])
plt.ylabel("Average Death Rate per 100k")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/pdmp_vs_deathrate.png")
plt.close()

# ========== 4. Naloxone Access Law compare ==========
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='naloxone_access', y='death_rate_per_100k', estimator='mean')
plt.title("Average Death Rate by Naloxone Access Law")
plt.xticks([0, 1], ["No Access", "Access Law in Place"])
plt.ylabel("Average Death Rate per 100k")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/naloxone_vs_deathrate.png")
plt.close()

# ========== 5. Medicaid Expansion ==========
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='medicaid_expansion', y='death_rate_per_100k', estimator='mean')
plt.title("Death Rate by Medicaid Expansion")
plt.xticks([0, 1], ["Not Expanded", "Expanded"])
plt.ylabel("Average Death Rate per 100k")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/medicaid_vs_deathrate.png")
plt.close()

# ========== 6. Heatmap of Annual Death Rates by State ==========
pivot_rate = data.pivot_table(index='state', columns='year', values='death_rate_per_100k')
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_rate, cmap="Blues", linewidths=0.3, linecolor='gray')
plt.title("Opioid Death Rate per 100k by State and Year")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/state_year_rate_heatmap.png")
plt.close()

# ========== 7. Income vs. Death Rate ==========
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='median_household_income', y='death_rate_per_100k', hue='year', alpha=0.6)
plt.title("Income vs. Opioid Death Rate")
plt.xlabel("Median Household Income")
plt.ylabel("Death Rate per 100k")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/income_vs_deathrate.png")
plt.close()

# ========== 8. Unemployment Rate vs. Death Rate (If Available) ==========
if 'unemployment_rate' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='unemployment_rate', y='death_rate_per_100k', hue='year', alpha=0.6)
    plt.title("Unemployment Rate vs. Opioid Death Rate")
    plt.xlabel("Unemployment Rate (%)")
    plt.ylabel("Death Rate per 100k")
    plt.tight_layout()
    plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/unemp_vs_deathrate.png")
    plt.close()

# ========== 9. Feature Correlation Heatmap ==========
corr_features = [
    'death_rate_per_100k',
    'pdmp_implemented',
    'naloxone_access',
    'poverty_population',
    'median_household_income',
    'medicaid_expansion'
]
if 'unemployment_rate' in data.columns:
    corr_features.append('unemployment_rate')

corr_matrix = data[corr_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/feature_correlation.png")
plt.close()

print("✅ All upgraded EDA plots saved to /outputs/")

# ========== Poverty vs. Death Rate ==========
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='poverty_population', y=data['deaths'] / data['population'] * 100000,
                hue='year', palette='coolwarm', alpha=0.7)
plt.title("Poverty Population vs. Opioid Death Rate")
plt.xlabel("Poverty Population")
plt.ylabel("Death Rate per 100k")
plt.tight_layout()
plt.savefig("/Users/zhuxumo/Desktop/opioid-project/outputs/poverty_vs_deathrate.png")
plt.close()
