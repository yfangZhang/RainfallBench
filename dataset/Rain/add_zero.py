import pandas as pd
import numpy as np

# 读取数据（假设数据已加载为 DataFrame，列名为 'tp'）
# 如果是从 CSV 读取：
df = pd.read_csv('JFNG_data_15min.csv', sep=',')
print(df.columns)  # 查看 DataFrame 的所有列名
# print(df)
# # 确保按时间排序（如果未排序）
# df = df.sort_values('date')

# 计算前3、5、7个时间点的零值（1e-5）数量
# 短期窗口（3,6,12个时间点，对应0.75h,1.5h,3h）
windows = [4,6]  # 24个点=6小时（假设15分钟间隔）
for w in windows:
    df[f'zero_count_last_{w}'] = df['tp'].rolling(w, min_periods=1).apply(lambda x: (x == 1e-5).sum())
# 24小时零值频率（假设数据为15分钟间隔，24h=96个点）
# df['zero_freq_24h'] = df['tp'].rolling(96, min_periods=1).apply(lambda x: (x == 1e-5).mean())

# # 7天零值频率（7d=672个点，按实际数据调整）
# df['zero_freq_7d'] = df['tp'].rolling(672, min_periods=1).apply(lambda x: (x == 1e-5).mean())
# # 计算连续零值区块（单位：时间点个数）
# df['dry_blocks'] = (df['tp'] != 1e-5).cumsum()

# # 仅在干旱时段（tp=1e-5）计算PWV均值
# df['pwv_during_dry'] = np.where(
#     df['tp'] == 1e-5,
#     df['pwv'].rolling(12, min_periods=1).mean(),
#     np.nan
# )

# 填充缺失值（如初始窗口不足）
df = df.fillna(method='ffill').fillna(0)
# 显示结果
cols = df.columns.tolist()
cols.remove('tp')
cols.append('tp')
df = df[cols]

# 保存结果（可选）
df.to_csv('weather_data_with_zero_counts8.csv', index=False)