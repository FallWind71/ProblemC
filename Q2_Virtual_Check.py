import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import os

# 读取每日品类加权成本表，包含价格和销量
df = pd.read_excel("每日品类加权成本.xlsx")

# 假设'加权单位售价'是价格列，'总销量'是销量列
price_col = '加权单位售价'
sales_col = '总销量'

# 创建输出文件夹
os.makedirs("每品类价格-销量_等腕回归", exist_ok=True)

sns.set(style="whitegrid", font="WenQuanYi Zen Hei", font_scale=1.2)

# 对每个品类进行等腕回归，并绘图
for category, data in df.groupby('分类名称'):
    # 取价格和销量，按价格排序
    df_cat = data.dropna(subset=[price_col, sales_col]).sort_values(price_col)
    x = df_cat[price_col].values
    y = df_cat[sales_col].values
    
    # 等腕回归，强制销量随价格单调递减
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
    y_iso = iso.fit_transform(x, y)
    
    # 绘制散点和回归曲线
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, label='原始数据', alpha=0.5, s=20)
    plt.plot(x, y_iso, color='red', linewidth=2, label='等腕回归 (单调递减)')
    plt.title(f"{category}：价格 vs 销量（等腕回归平滑）")
    plt.xlabel("价格（元）")
    plt.ylabel("销量（千克）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"每品类价格-销量_等腕回归/{category}_price_sales_isotonic.png")
    plt.close()

print("所有品类价格-销量等腕回归图已保存到 ./每品类价格-销量_等腕回归/")
