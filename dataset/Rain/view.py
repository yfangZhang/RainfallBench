import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取CSV文件（适配15分钟间隔数据）
df = pd.read_csv('./JFNG_data_15min.csv', 
                parse_dates=['date'], 
                dayfirst=False)
df.set_index('date', inplace=True)

# 检查数据完整性
print(f"原始数据时间范围：{df.index.min()} 至 {df.index.max()}")
print(f"总数据点数：{len(df)}，理论应有数据点：{(df.index.max()-df.index.min()).total_seconds()/900}")

# 创建可视化画布
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
plt.subplots_adjust(hspace=0.45)

# 原始15分钟数据可视化（修改点1）
ax1.plot(df.index, df['tp'], 
        marker='o',          # 添加数据点标记
        markersize=4, 
        linestyle='--',      # 虚线连接
        linewidth=0.8,
        color='#1f77b4')
ax1.set_title('15-Minute Interval Raw Data', pad=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # 每3小时一个主刻度
ax1.xaxis.set_minor_locator(mdates.HourLocator())            # 每小时次刻度
ax1.grid(which='both', linestyle=':', alpha=0.6)

# 小时均值（自动对齐时间戳）
hourly = df['tp'].resample('H').mean()
ax2.bar(hourly.index, hourly, 
       width=0.03,          # 调整柱宽适应小时刻度
       color='#ff7f0e',
       edgecolor='black')
ax2.set_title('Hourly Average', pad=15)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 每天一个主刻度

# 日均值（带缺失值检查）
daily = df['tp'].resample('D').mean()
print("每日数据量统计：")
print(df['tp'].resample('D').count())  # 检查每日数据点数量

ax3.plot(daily.index, daily, 
        marker='s',         # 方形标记
        markersize=6,
        linewidth=1.5,
        color='#2ca02c')
ax3.set_title('Daily Average', pad=15)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1)) 

# 公共格式设置
for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='x', rotation=35)
    ax.set_ylabel('TP Value', labelpad=10)

fig.suptitle('Multi-Scale Temporal Analysis (15-min Raw Data)', 
            y=0.95, 
            fontsize=14,
            fontweight='bold')

# 保存高分辨率图片
plt.savefig('multi_scale_15min.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white')

plt.show()