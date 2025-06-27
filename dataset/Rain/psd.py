import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import matplotlib

# 使用 SVG 输出
matplotlib.use("svg")

# 读取数据
df = pd.read_csv("JFNG_data_15min.csv")

# 确保 tp 列存在并非空
tp_series = df['tp'].dropna().values

# 零均值化（移除直流分量）
tp_centered = tp_series - np.mean(tp_series)

# 执行傅里叶变换
fft_result = fft(tp_centered)
power_spectrum = np.abs(fft_result[:len(fft_result)//2])**2  # 取前半段
frequencies = np.fft.fftfreq(len(tp_centered), d=1)[:len(fft_result)//2]

plt.figure(figsize=(10, 4))
plt.plot(frequencies[1:], power_spectrum[1:])  # 跳过频率0（直流分量）
plt.title("Power Spectrum of TP",fontsize=16)
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.grid(True)
plt.tight_layout()
plt.savefig("tp_spectrum.svg", format='svg')  # 输出 SVG 图
plt.show()
