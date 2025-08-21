import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== 路径设置 ==========
DATA_PATH = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv"
OUTPUT_FOLDER = "/Users/zhuxumo/Desktop/opioid-project/outputs/"

# ========== 读取数据 ==========
df = pd.read_csv(DATA_PATH)

# 生成死亡率变量
df['death_rate_per_100k'] = df['deaths'] / df['population'] * 100000

# one-hot 编码 state
df = pd.get_dummies(df, columns=['state'], drop_first=True)

# ========== 特征设置 ==========
features = [
    'pdmp_implemented',
    'naloxone_access',
    'poverty_population',
    'population',
    'median_household_income',
    'medicaid_expansion'
] + [col for col in df.columns if col.startswith('state_')]

# ========== 划分训练与测试集 ==========
train_data = df[df['year'] < 2019]  # 使用2010–2018训练
test_data = df[df['year'] >= 2019]  # 预测2019、2020

X_train = train_data[features]
y_train = train_data['death_rate_per_100k']
X_test = test_data[features]
y_test = test_data['death_rate_per_100k']

# ========== 模型训练 ==========
model = xgb.XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# ========== 预测与评估 ==========
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"🟩 [Income + Medicaid Expansion] MSE: {mse:.2f}, R^2: {r2:.3f}")

# ========== 预测 vs 实际图（按年份分图）==========
test_data = test_data.copy()
test_data['y_true'] = y_test
test_data['y_pred'] = y_pred

original_raw = pd.read_csv(DATA_PATH)
test_data['state'] = original_raw.loc[test_data.index, 'state']
test_data['year'] = original_raw.loc[test_data.index, 'year']

for yr in [2019, 2020]:
    plt.figure(figsize=(10, 8))
    sub = test_data[test_data['year'] == yr]
    sns.scatterplot(data=sub, x='y_true', y='y_pred', hue='state',
                    palette='tab20', s=90, edgecolor='white', alpha=0.85)
    plt.plot([sub['y_true'].min(), sub['y_true'].max()],
             [sub['y_true'].min(), sub['y_true'].max()],
             '--', color='gray')
    plt.xlabel("Actual Death Rate per 100k")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual ({yr})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"predicted_vs_actual_{yr}.png"))
    plt.close()

# ========== 特征重要性图 ==========
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "feature_importance.png"))
plt.close()

print("✅ All upgraded visual outputs saved!")
