from sbc import sbc_class
import pandas as pd

df = pd.read_csv("./JFNG_data_15min.csv")
print(df.iloc[:, -1].replace(1e-5, 0))
## 1 target
out1 = sbc_class.sbc_class(df.iloc[:, -1].replace(1e-5, 0), plot_type = 'summary')
print(out1)