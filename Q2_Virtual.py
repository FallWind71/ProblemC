import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# 生成拟合函数公式字符串
def format_equation(name, params):
    if name == "线性":
        a, b = params
        return f"y = {a:.4f}·x + {b:.4f}"
    elif name == "对数":
        a, b = params
        return f"y = {a:.4f}·ln(x) + {b:.4f}"
    elif name == "幂函数":
        a, b = params
        return f"y = {a:.4f}·x^{b:.4f}"
    elif name == "指数":
        a, b = params
        return f"y = {a:.4f}·e^({b:.4f}·x)"
    elif "多项式" in name:
        terms = [f"{p:.4f}·x^{i}" for i, p in reversed(list(enumerate(params)))]
        return "y = " + " + ".join(terms).replace("+ -", "- ")
    else:
        return "未知函数"


# 打印公式


# 设置中文和图表风格
sns.set(style="whitegrid", font="WenQuanYi Zen Hei", font_scale=1.2)
os.makedirs("每品类售价销量关系图", exist_ok=True)

# 读取数据
df = pd.read_excel("每日品类加权成本.xlsx")
print("每日品类加权成本.xlsx读取完成\n")

# 定义拟合函数们
def linear(x, a, b):
    return a * x + b

def log_func(x, a, b):
    return a * np.log(x + 1e-5) + b  # 避免 log(0)

def power_func(x, a, b):
    return a * np.power(x, b)

def expo_func(x, a, b):
    return a * np.exp(b * x)

    # 添加多项式拟合函数（如二次、三次、四次）
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def poly4(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


fit_funcs = {
    "线性": linear,
    "对数": log_func,
    "幂函数": power_func,
    "指数": expo_func,
    "二次多项式": poly2,
    "三次多项式": poly3,
    "四次多项式": poly4,
}

# 遍历每个品类
for category in df['分类名称'].unique():
    data = df[df['分类名称'] == category].copy()
    x = data['加权单位售价'].values
    y = data['总销量'].values
        # 去除离群值（基于 IQR 方法）
    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    lower_y = q1_y - 1.5 * iqr_y
    upper_y = q3_y + 1.5 * iqr_y

    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    lower_x = q1_x - 1.5 * iqr_x
    upper_x = q3_x + 1.5 * iqr_x

    mask = (x >= lower_x) & (x <= upper_x) & (y >= lower_y) & (y <= upper_y)
    x = x[mask]
    y = y[mask]

    df_filtered = pd.DataFrame({'加权单位售价': x, '总销量': y})
    # ✨ 将 x 按 0.2 间隔划分区间，计算每段的质心
    bin_width = 0.5
    x_bins = np.arange(x.min(), x.max() + bin_width, bin_width)
    bin_indices = np.digitize(x, x_bins)

    # 用 DataFrame 聚合每个 bin 的中心点
    binned = pd.DataFrame({'售价': x, '销量': y, 'bin': bin_indices})
    grouped = binned.groupby('bin').agg({
        '售价': 'mean',
        '销量': 'mean'
    }).reset_index()

    # 替换为质心数据
    x = grouped['售价'].values
    y = grouped['销量'].values
    print(f"🎯 Binning后质心点数：{len(x)}\n")

    print(f"聚合后点数（售价唯一值数量）：{len(x)}")
    print(f"剔除离群点后数据量：{len(x)}")
    original_n = len(data)
    filtered_n = len(x)
    print(f"📉 离群点剔除：原始 {original_n} 条，剩余 {filtered_n} 条（{(filtered_n/original_n)*100:.1f}%）")




    # 输出基本统计数据
    corr, _ = pearsonr(x, y)
    print(f"👉 分类：{category}")
    print(f"皮尔逊相关系数：{corr:.4f}")
    print(f"x（售价）范围：{x.min():.2f} ~ {x.max():.2f}")
    print(f"y（销量）范围：{y.min():.2f} ~ {y.max():.2f}")

    # 拟合选择：尝试每种函数并记录 R²
    best_func = None
    best_name = ""
    best_r2 = -np.inf
    best_params = None

    for name, func in fit_funcs.items():
        try:
            params, _ = curve_fit(func, x, y, maxfev=10000)
            y_pred = func(x, *params)
            r2 = r2_score(y, y_pred)
            print(f"  {name}拟合 R²: {r2:.4f}")
            if r2 > best_r2:
                best_r2 = r2
                best_func = func
                best_name = name
                best_params = params
        except:
            print(f"  {name}拟合失败")
        if best_func is not None:
            equation = format_equation(best_name, best_params)
            print(f"📐 最佳拟合公式（{best_name}）：{equation}\n")


    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 原始数据点（透明显示，表达数据分布）
    sns.scatterplot(x=data['加权单位售价'].values, y=data['总销量'].values,
                    s=20, color='blue', alpha=0.1, label="原始数据点")

    # 质心点（可见）
    sns.scatterplot(x=x, y=y, s=40, color='blue', label="售价区间质心点")

    # 拟合曲线（基于质心）
    if best_func is not None:
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = best_func(x_fit, *best_params)
        plt.plot(x_fit, y_fit, color='red', label=f"{best_name}拟合 R²={best_r2:.3f}")

    plt.title(f"{category}：售价 vs 销量")
    plt.xlabel("加权单位售价（元/千克）")
    plt.ylabel("总销量（千克）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"每品类售价销量关系图/{category}_售价_vs_销量_拟合图-质心.png")
    plt.close()

    print(f"✅ 图像保存完成：{category}_售价_vs_销量_拟合图-质心.png\n")

