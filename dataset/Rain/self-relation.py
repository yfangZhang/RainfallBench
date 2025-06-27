# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 读取CSV文件
# file_path = 'JFNG_data_15min.csv'  # 替换为你的CSV文件路径
# data = pd.read_csv(file_path)

# # 选择后6列
# last_6_columns = data.iloc[:, -6:]

# # 计算相关性矩阵
# correlation_matrix = last_6_columns.corr(method='spearman') #'pearson', 'kendall', 'spearman'

# # 绘制相关性矩阵
# plt.figure(figsize=(10, 8))
# heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Correlation Matrix(Spearman Method)')

# # 保存为图片
# output_path = 'correlation_matrix_spearman.pdf'  # 替换为你想要保存的图片路径
# plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存图片，设置分辨率和边框

# # 显示图像
# plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 读取CSV文件
file_path = 'JFNG_data_15min.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 选择后6列
last_6_columns = data.iloc[:, -6:]

# 相关性方法列表
methods = ['pearson', 'kendall', 'spearman']

# 设置输出PDF路径
output_pdf_path = 'correlation_matrices_horizontal.pdf'

# 创建PDF文件
with PdfPages(output_pdf_path) as pdf:
    # 创建一行三列的子图
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))  # 宽度设置为3倍单图宽度

    for ax, method in zip(axes, methods):
        # 计算相关性矩阵
        corr_matrix = last_6_columns.corr(method=method)

        # 绘制热力图
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, cbar=False)
        ax.set_title(f'{method.capitalize()}',fontsize=30)
        
        # 增大坐标轴标签字体大小
        ax.tick_params(axis='both', which='major', labelsize=16)

    # 调整布局并保存为一页PDF
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"相关性热图已横向排列并保存到：{output_pdf_path}")
