import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# 路径
DATA_PATH = "/Users/zhuxumo/Desktop/opioid-project/datasets/merged_data_final.csv"
OUTPUT_TXT = "/Users/zhuxumo/Desktop/opioid-project/outputs/policy_regression_summary.txt"

# 读取数据
data = pd.read_csv(DATA_PATH)
data = data.set_index(['state', 'year'])

# 构造死亡率作为因变量
data['death_rate'] = data['deaths'] / data['population'] * 100000

# 标准化连续变量
data['poverty_scaled'] = (data['poverty_population'] - data['poverty_population'].mean()) / data['poverty_population'].std()
data['income_scaled'] = (data['median_household_income'] - data['median_household_income'].mean()) / data['median_household_income'].std()

# 构造回归模型
exog_vars = ['pdmp_implemented', 'naloxone_access', 'medicaid_expansion', 'poverty_scaled', 'income_scaled']
exog = sm.add_constant(data[exog_vars])

model = PanelOLS(dependent=data['death_rate'], exog=exog, entity_effects=True, time_effects=True)
results = model.fit(cov_type='clustered', cluster_entity=True)

# 输出
with open(OUTPUT_TXT, 'w') as f:
    f.write(str(results.summary))

print("✅ policy regression updated and saved.")
