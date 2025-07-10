import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
sns.set(style="whitegrid", font="WenQuanYi Zen Hei", font_scale=1.2)
# 创建文件夹保存图像
os.makedirs("每品类成本傅立叶拟合图", exist_ok=True)
df = pd.read_excel("每日品类加权成本.xlsx")
print("每日品类加权成本.xlsx读取完成\n")

# 转换为天序号
df['天序号'] = (df['销售日期'] - df['销售日期'].min()).dt.days

# 傅立叶函数
def fourier_series(x, *a):
    n = (len(a) - 1) // 2
    result = a[0]
    for i in range(1, n + 1):
        result += a[2 * i - 1] * np.sin(2 * np.pi * i * x / 365.25) + a[2 * i] * np.cos(2 * np.pi * i * x / 365.25)
    return result

# 阶数
N = 7
initial_guess = [0.0] * (2 * N + 1)

# 拟合每个分类
for category in df['分类名称'].unique():
    data = df[df['分类名称'] == category].copy()
    x = data['天序号'].values
    y = data['加权单位成本'].values

    try:
        params, _ = curve_fit(fourier_series, x, y, p0=initial_guess, maxfev=10000)
        y_pred = fourier_series(x, *params)
        r2 = r2_score(y, y_pred)
        residuals = y - y_pred
        std_residual = np.std(residuals)
        mean_y = np.mean(y)
        amp_y = y.max() - y.min()
        amp_pred = y_pred.max() - y_pred.min()

        # 调试输出
        print(f"👉 分类：{category}")
        print(f"R² = {r2:.4f}")
        print(f"原始均值：{mean_y:.4f}，原始波动幅度：{amp_y:.4f}")
        print(f"拟合波动幅度：{amp_pred:.4f}")
        print(f"残差标准差：{std_residual:.4f}\n")

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

        # 可选保存残差图（开启下面注释即可）
        plt.figure(figsize=(10, 3))
        plt.plot(data['销售日期'], residuals, color='orange', label="残差")
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f"{category}：残差图")
        plt.tight_layout()
        plt.savefig(f"每品类成本傅立叶拟合图/{category}_残差图.png")
        plt.close()

        print(f"✅ 图像保存完成：{category}_加权成本_傅立叶拟合.png\n")

    except Exception as e:
        print(f"❌ {category} 拟合失败：{e}")

