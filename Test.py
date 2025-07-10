import pandas as pd
import numpy as np

df3 = pd.read_excel('附件3.xlsx')  # 批发价记录（按日期+编码）
print("附件三读取完成\n")

df3.info()