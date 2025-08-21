import pandas as pd
import statsmodels.formula.api as smf

# 设置路径
DATA_PATH = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv"
OUTPUT_TXT = "/Users/zhuxumo/Desktop/opioid-project/outputs/mixed_effects_summary.txt"

# 读取数据
data = pd.read_csv(DATA_PATH)

# 添加死亡率
data['death_rate'] = data['deaths'] / data['population'] * 100000

# 标准化贫困人口变量
data['poverty_scaled'] = (data['poverty_population'] - data['poverty_population'].mean()) / data['poverty_population'].std()

# Mixed Effects Model（随机效应组为 state）
model = smf.mixedlm(
    formula="death_rate ~ pdmp_implemented + naloxone_access + medicaid_expansion + unemployment_rate + poverty_scaled",
    data=data,
    groups=data["state"]
)

results = model.fit()
print(results.summary())

# 保存结果
with open(OUTPUT_TXT, 'w') as f:
    f.write(str(results.summary()))

print("✅ Mixed Effects Regression Summary saved!")
