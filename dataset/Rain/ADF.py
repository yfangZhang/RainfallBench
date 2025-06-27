import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 1. 读取CSV文件（假设文件名为data.csv，时间序列列名为'tp'）
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'tp' not in df.columns:
            raise ValueError("列 'tp' 不存在于文件中")
        return df['tp'].dropna()  # 移除缺失值
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None

# 2. 执行ADF检验并打印结果
def adf_test(series, significance_level=0.05):
    result = adfuller(series, autolag='AIC')
    print('==== ADF检验结果 ====')
    print(f'ADF统计量: {result[0]:.4f}')
    print(f'p值: {result[1]:.4f}')
    print(f'临界值:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.4f}')
    
    # 判断是否平稳
    if result[1] < significance_level:
        print(f"结论: p值 < {significance_level}，拒绝原假设，序列平稳")
    else:
        print(f"结论: p值 >= {significance_level}，无法拒绝原假设，序列非平稳")

# 3. 可视化时间序列
def plot_series(series):
    plt.figure(figsize=(12, 6))
    series.plot()
    plt.title('时间序列可视化 (tp列)')
    plt.xlabel('时间/索引')
    plt.ylabel('值')
    plt.grid(True)
    plt.show()

# 主流程
if __name__ == '__main__':
    # 替换为你的CSV文件路径
    file_path = './JFNG_data_15min.csv'
    series = load_data(file_path)
    
    if series is not None:
        # 打印数据摘要
        print(f"数据摘要:\n{series.describe()}\n")
        
        # 可视化
        plot_series(series)
        
        # ADF检验
        adf_test(series)