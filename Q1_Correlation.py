import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", font="WenQuanYi Zen Hei", font_scale=1.2)
# 读取数据
df = pd.read_excel("品类季度销量.xlsx")

# 创建季度标签用于行索引
df['季度标签'] = df['年'].astype(str) + '-Q' + df['季度'].astype(str)

# 生成透视表：每行一个季度，每列一个品类，值为销量
pivot_df = df.pivot_table(
    index='季度标签',
    columns='分类名称',
    values='销量(千克)',
    aggfunc='sum'
).fillna(0)

# 计算皮尔逊相关系数
corr = pivot_df.corr()

# 热力图可视化
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("各蔬菜品类季度销量相关性热力图")
plt.tight_layout()
plt.savefig("各品类销量相关性热力图.png")
plt.show()
