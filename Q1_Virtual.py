import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 读取我们统计好的表格
df = pd.read_excel("品类季度销量.xlsx")

# 创建 "季度标签" 用于 x 轴
df['季度标签'] = df['年'].astype(str) + '-Q' + df['季度'].astype(str)
df = df.sort_values(['年', '季度'])

# 设置画图风格和字体（防止中文乱码）
#这个字体你不一定有，如果你尝试运行失败，可以换成Sim Hei
sns.set(style="whitegrid", font="WenQuanYi Zen Hei", font_scale=1.2)

# 画图,尺寸为14,7
plt.figure(figsize=(14, 7))

# ❗️关键：每条折线表示一个品类（hue参数）
sns.barplot(
    data=df,
    x='季度标签',
    y='销量(千克)',
    hue='分类名称',  # 每个品类一条线
)

plt.title("各蔬菜品类季度销量趋势图")
plt.xlabel("季度")
plt.ylabel("销量（千克）")

#横坐标的名字的旋转角度
plt.xticks(rotation=40)

plt.legend(title="品类")

#自动规划布局
plt.tight_layout()

plt.savefig("各品类季度销量趋势图.png")
plt.show()
