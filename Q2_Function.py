import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# 创建文件夹保存图像
os.makedirs("每品类成本傅立叶拟合图", exist_ok=True)
df = pd.read_excel("每日品类加权成本.xlsx")
print("每日品类加权成本.xlsx读取完成\n")
# 将日期转为数值时间（以最早日期为起点）
df['天序号'] = (df['销售日期'] - df['销售日期'].min()).dt.days

# 定义傅立叶拟合函数（N阶）
def fourier_series(x, *a):
    n = (len(a) - 1) // 2
    result = a[0]
    for i in range(1, n + 1):
        result += a[2 * i - 1] * np.sin(2 * np.pi * i * x / 365.25) + a[2 * i] * np.cos(2 * np.pi * i * x / 365.25)
    return result

# 设置傅立叶阶数
N = 3  # 3阶傅立叶展开（可根据需要调整）
initial_guess = [0.0] * (2 * N + 1)

# 遍历每个分类
for category in df['分类名称'].unique():
    data = df[df['分类名称'] == category].copy()
    x = data['天序号'].values
    y = data['加权单位成本'].values

    try:
        params, _ = curve_fit(fourier_series, x, y, p0=initial_guess, maxfev=10000)
        y_pred = fourier_series(x, *params)
        r2 = r2_score(y, y_pred)

        # 画图
        plt.figure(figsize=(12, 5))
        plt.plot(data['销售日期'], y, label='原始数据', color='gray', alpha=0.5)
        plt.plot(data['销售日期'], y_pred, label=f'傅立叶拟合 (R²={r2:.3f})', color='red')
        plt.title(f"{category}：加权单位成本随时间变化（傅立叶{N}阶拟合）")
        plt.xlabel("日期")
        plt.ylabel("加权单位成本（元/千克）")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"每品类成本傅立叶拟合图/{category}_加权成本_傅立叶拟合.png")
        plt.close()

        print(f"✅ 成功拟合并保存：{category}_加权成本_傅立叶拟合.png")

    except Exception as e:
        print(f"❌ {category} 拟合失败：{e}")
