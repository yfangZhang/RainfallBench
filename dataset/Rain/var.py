import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置更鲜艳的莫兰蒂色系
morandi_palette = [
    "#5A7D7C",  # 柔和青蓝
    "#D97D6A",  # 暖珊瑚红
    "#8B7D8B",  # 柔紫灰
    "#C76F51",  # 暖橙棕
    "#C8B552",  # 淡芥末黄
    "#4B6473"   # 深石板蓝 
]

# 标注标签 a-f
labels = ['a', 'b', 'c', 'd', 'e', 'f']

# 读取CSV文件
file_path = 'JFNG_data_15min.csv'  # 替换为你的路径
df = pd.read_csv(file_path)

# 获取最后6个变量
variables = df.columns[-6:]
data = df[variables]

# 创建3x2子图布局
fig, axes = plt.subplots(3, 2, figsize=(12, 10), facecolor="white")
axes = axes.flatten()
axis_label_fontsize = 12
tick_label_fontsize = 14
# 绘图
for i, var in enumerate(variables):
    ax = axes[i]
    
    if i == 5:  # tp 的特殊处理
        tp_data = data[var]
        zero_count = (tp_data == 1e-5).sum()
        nonzero_data = tp_data[tp_data > 1e-5]

        # 绘制非零分布的直方图
        sns.histplot(nonzero_data, ax=ax, color=morandi_palette[i],
                     kde=False, edgecolor='none', bins=30)

        # 添加一个 bar 表示 0 值数量
        ax.bar(x=0, height=zero_count, width=0.8, color=morandi_palette[i], label='tp = 0')

        ax.set_title(f'Distribution of {var}', fontsize=22, color="#4a4a4a")
        ax.set_xlabel(f'{var} value')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_xlabel(f'{var} value', fontsize=axis_label_fontsize)
        ax.set_ylabel('Count', fontsize=axis_label_fontsize)
        ax.legend(fontsize=tick_label_fontsize)
        ax.set_xlim(left=-2)  # 给0留位置显示
    else:
        sns.histplot(data[var], ax=ax, color=morandi_palette[i],
                     kde=True, edgecolor='none')
        ax.set_title(f'Distribution of {var}', fontsize=22, color="#4a4a4a")
        ax.set_xlabel(var, fontsize=axis_label_fontsize)
        ax.set_ylabel('Density', fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    ax.set_facecolor("white")
    ax.tick_params(colors="#4a4a4a")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.05, 1.1, f'({labels[i]})', transform=ax.transAxes,
             fontsize=18, fontweight='bold', color="#333333")

# 删除多余子图（如果不足6个变量）
for j in range(len(variables), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# 保存为矢量 PDF
plt.savefig("variable_distributions_tp_with_zero.pdf", format='pdf', bbox_inches='tight')
plt.show()
