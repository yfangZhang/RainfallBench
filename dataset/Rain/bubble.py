import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('JFNG_data_15min.csv')  # 替换为你的文件路径

# 获取最后一列数据
last_column = df.iloc[:, -1]

# 统计值的出现次数
value_counts = last_column.value_counts().sort_index()

# 准备数据
x = value_counts.index
y = value_counts.values
sizes = y * 50  # 放大气泡大小

# 莫兰蒂粉色
color = '#E0B0FF'

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=sizes, alpha=0.6, c=color, edgecolors='black')

plt.title('Bubble Chart of Last Column Frequencies')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存为 PDF
plt.savefig('bubble_chart.pdf', format='pdf', bbox_inches='tight')
plt.show()
