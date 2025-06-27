import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("JFNG_data_15min.csv")

# 转换时间列（你可以调整时间列名）
df['date'] = pd.to_datetime(df['date'])

# 提取小时（0~23）
df['hour'] = df['date'].dt.hour

# 计算每小时平均 tp
hourly_mean = df.groupby('hour')['tp'].mean()

# 设置莫兰蒂风格颜色
color = "#FFD88C"

# 绘图
plt.figure(figsize=(8, 4))
sns.lineplot(x=hourly_mean.index, y=hourly_mean.values, color=color, marker="o")
plt.title("Average tp by Hour of Day", fontsize=14)
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Mean tp", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(range(0, 24))
plt.tight_layout()

# 保存为 PDF（矢量图适合 PPT 插入）
plt.savefig("tp_hourly_mean.svg", format="svg")
plt.show()
