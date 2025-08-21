import re
import pandas as pd

# 输入和输出路径
txt_file = "/Users/zhuxumo/Desktop/opioid-project/outputs/mixed_effects_summary.txt"
csv_file = "/Users/zhuxumo/Desktop/opioid-project/datasets/mixed_effects_summary.csv"

# 读取 txt 文件
with open(txt_file, "r") as f:
    text = f.read()

# 用正则匹配：变量名  系数  p值
pattern = re.compile(r"(\w+)\s+([\-0-9.]+)\s+[0-9.Ee\-]+\s+([0-9.Ee\-]+)")
matches = pattern.findall(text)

# 转成 DataFrame
df = pd.DataFrame(matches, columns=["variable", "coef", "pvalue"])
df["coef"] = df["coef"].astype(float)
df["pvalue"] = df["pvalue"].astype(float)

# 保存为 CSV
df.to_csv(csv_file, index=False)
print(f"✅ 提取完成，已保存为 {csv_file}")
print(df)
