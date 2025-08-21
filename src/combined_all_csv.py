import os
import pandas as pd

# 设置路径
input_folder = '/Users/zhuxumo/Desktop/opioid-project/datasets/Unemployment_rate_csv'   # 替换为你CSV文件夹路径
output_file = '/Users/zhuxumo/Desktop/opioid-project/datasets/unemploymentRate_cleaned.csv'  # 输出文件名

# 空列表储存结果
all_data = []

for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        file_path = os.path.join(input_folder, file)
        try:
            # 读取文件，跳过无效的前几行（找出“Year”行的位置）
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "Year" in line and "Annual" in line:
                        start_row = i + 1
                        break
                else:
                    continue  # 如果没找到有效行，跳过文件
            
            # 从有效行开始读取数据
            df = pd.read_csv(file_path, skiprows=start_row - 1)
            df = df.dropna()  # 去除空行
            df.columns = ['Year', 'Unemployment Rate']  # 重命名列

            # 提取州名
            with open(file_path, 'r') as f:
                content = f.read()
                for line in content.splitlines():
                    if "Area:" in line:
                        state_name = line.split(",")[-1].strip()
                        break
                else:
                    state_name = "Unknown"

            # 添加州名列
            df['State'] = state_name

            # 加入总数据列表
            all_data.append(df)

        except Exception as e:
            print(f"❌ 文件处理失败: {file}, 错误: {e}")

# 合并所有数据
merged_df = pd.concat(all_data, ignore_index=True)

# 调整列顺序
merged_df = merged_df[['State', 'Year', 'Unemployment Rate']]

# ✅ 重命名列名为小写
merged_df = merged_df.rename(columns={
    'State': 'state',
    'Year': 'year',
    'Unemployment Rate': 'unemployment_rate'
})

# 保存为新的CSV
merged_df.to_csv(output_file, index=False)
print(f"✅ 已保存合并文件到: {output_file}")


