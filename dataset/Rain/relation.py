import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'JFNG_data_15min.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 选择后6列
last_6_columns = data.iloc[:, -6:]

# 选择目标变量和特征变量
X = last_6_columns.iloc[:, :-1]  # 特征变量（组合变量）
y = last_6_columns.iloc[:, -1]   # 目标变量（单变量）

# 拟合线性回归模型
model = LinearRegression()
model.fit(X, y)

# 获取回归系数
coefficients = pd.DataFrame(model.coef_, index=X.columns, columns=['Coefficient'])

# 绘制回归系数热力图
plt.figure(figsize=(10, 8))
sns.heatmap(coefficients, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Regression Coefficients for Single Variable')
output_path = 'Regression Coefficients.png'  # 替换为你想要保存的图片路径
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存图片，设置分辨率和边框
plt.show()