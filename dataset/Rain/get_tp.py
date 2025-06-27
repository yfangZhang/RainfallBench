import pandas as pd

# 读取CSV文件
df = pd.read_csv("JFNG_data_15min.csv")

# 仅保留date和tp列
df = df[["date", "pwv","tp"]]

# 保存到新文件
df.to_csv("JFNG_data_15min_tp&pwv_only.csv", index=False)