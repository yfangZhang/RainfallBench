import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv('JFNG_data_15min.csv')
tp_series = df['tp'].dropna()

fig, ax = plt.subplots(figsize=(10,5))
acf_plot = plot_acf(tp_series, lags=50, alpha=0.05, ax=ax)

for bar in ax.patches:
    bar.set_facecolor('#C5CAE1')

plt.title('Autocorrelation of tp column')
plt.ylim(0, 1)
ax.set_xlabel('Lag (time steps)')
ax.set_ylabel('Autocorrelation coefficient')
# 保存为SVG文件
plt.savefig('tp_acf_plot.svg', format='svg',bbox_inches='tight')

plt.show()
