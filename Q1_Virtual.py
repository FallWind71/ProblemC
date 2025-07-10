import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.dates as mdates

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

#####################################################################
# 读取每日成本数据
df_Cost = pd.read_excel("每日品类加权成本.xlsx")
df_Cost.info()
# 创建标准 datetime 列
df_Cost['销售日期'] = pd.to_datetime(
    df_Cost[['年', '月', '日']].rename(columns={'年': 'year', '月': 'month', '日': 'day'})
)

df_Cost = df_Cost.sort_values('销售日期')

# 创建输出文件夹
os.makedirs("每品类加权成本散点图", exist_ok=True)

# 设置图像风格
sns.set(style="whitegrid", font="WenQuanYi Zen Hei", font_scale=1.2)

# 遍历每个分类，单独画图
for category in df_Cost['分类名称'].unique():
    data = df_Cost[df_Cost['分类名称'] == category]

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='销售日期', y='加权单位成本', s=20, color='blue')  # ⬅ 点小一点

    plt.title(f"{category} 每日加权单位成本散点图")
    plt.xlabel("销售日期")
    plt.ylabel("加权单位成本")

    # 设置横轴格式：只显示每年一标
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())   # 每年一个主刻度
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 格式为年份

    plt.xticks(rotation=0)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f"每品类加权成本散点图/{category}_加权成本散点图.png")
    plt.close()





