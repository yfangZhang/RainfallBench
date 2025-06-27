import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# 读取CSV文件
df = pd.read_csv('./JFNG_data_15min.csv')  # 替换为您的文件路径

# 检查'tp'列是否存在
if 'tp' not in df.columns:
    raise ValueError("CSV文件中没有'tp'列")

# 泊松变换函数
def poisson_transform(x):
    return np.sqrt(x + 3/8)  # 常用的一种泊松变换(Anscombe变换)

# 创建变换器
poisson_transformer = FunctionTransformer(poisson_transform)

# 应用变换
df['tp_transformed'] = poisson_transformer.transform(df['tp'])

# 保存结果到新文件
df.to_csv('transformed_file.csv', index=False)

print("泊松变换完成，结果已保存到transformed_file.csv")