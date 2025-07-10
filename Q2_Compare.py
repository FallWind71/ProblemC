import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 读取数据 ==========
os.makedirs("拟合图像/傅立叶", exist_ok=True)
os.makedirs("拟合图像/Prophet", exist_ok=True)
os.makedirs("拟合图像/STL", exist_ok=True)



print("读取中...")
df = pd.read_excel("每日品类加权成本.xlsx")
df['天序号'] = (df['销售日期'] - df['销售日期'].min()).dt.days

# ========== 傅立叶函数 ==========
def fourier_series(x, *a):
    n = (len(a) - 1) // 2
    result = a[0]
    for i in range(1, n + 1):
        result += a[2 * i - 1] * np.sin(2 * np.pi * i * x / 365.25)
        result += a[2 * i] * np.cos(2 * np.pi * i * x / 365.25)
    return result

def fit_fourier(data, category, N=7):
    x = data['天序号'].values
    y = data['加权单位成本'].values
    p0 = [0.0] * (2 * N + 1)
    try:
        params, _ = curve_fit(fourier_series, x, y, p0=p0, maxfev=10000)
        y_pred = fourier_series(x, *params)
        r2 = r2_score(y, y_pred)
        res_std = np.std(y - y_pred)
        amp = y_pred.max() - y_pred.min()
        # 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(data['销售日期'], y, label='原始', alpha=0.5)
        plt.plot(data['销售日期'], y_pred, label='傅立叶拟合', color='red')
        plt.title(f"{category} - 傅立叶 R²={r2:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"拟合图像/傅立叶/{category}.png")
        plt.close()
        return r2, res_std, amp
    except:
        return np.nan, np.nan, np.nan

# ========== Prophet ==========
def fit_prophet(data, category):
    df_prophet = data[['销售日期', '加权单位成本']].rename(columns={
        '销售日期': 'ds', '加权单位成本': 'y'
    })
    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df_prophet)
        future = df_prophet[['ds']]
        forecast = model.predict(future)
        y_pred = forecast['yhat'].values
        y_true = df_prophet['y'].values
        r2 = r2_score(y_true, y_pred)
        res_std = np.std(y_true - y_pred)
        amp = y_pred.max() - y_pred.min()
        # 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(df_prophet['ds'], y_true, label='原始', alpha=0.5)
        plt.plot(df_prophet['ds'], y_pred, label='Prophet预测', color='green')
        plt.title(f"{category} - Prophet R²={r2:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"拟合图像/Prophet/{category}.png")
        plt.close()
        return r2, res_std, amp
    except:
        return np.nan, np.nan, np.nan

# ========== STL分解 ==========
def fit_stl(data, category):
    series = data.set_index('销售日期')['加权单位成本'].asfreq('D').interpolate()
    try:
        stl = STL(series, period=365)
        res = stl.fit()
        trend = res.trend
        r2 = r2_score(series, trend)
        res_std = np.std(series - trend)
        amp = trend.max() - trend.min()
        # 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(series.index, series.values, label='原始', alpha=0.5)
        plt.plot(series.index, trend.values, label='STL趋势', color='purple')
        plt.title(f"{category} - STL R²={r2:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"拟合图像/STL/{category}.png")
        plt.close()
        return r2, res_std, amp
    except:
        return np.nan, np.nan, np.nan

# ========== 主循环 ==========
results = []

for category in df['分类名称'].unique():
    data = df[df['分类名称'] == category].copy()
    # 去除加权单位成本的离群值（IQR 方法）
    q1 = data['加权单位成本'].quantile(0.25)
    q3 = data['加权单位成本'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    before = len(data)
    data = data[(data['加权单位成本'] >= lower_bound) & (data['加权单位成本'] <= upper_bound)]
    after = len(data)
    print(f"已剔除离群点：{before - after} 条，剩余 {after} 条数据")

    print(f"\n👉 分类：{category}")

    r2_f, std_f, amp_f = fit_fourier(data, category)
    r2_p, std_p, amp_p = fit_prophet(data, category)
    r2_s, std_s, amp_s = fit_stl(data, category)

    results.append({
        '分类': category,
        '方法': '傅立叶', 'R2': r2_f, '残差STD': std_f, '幅度': amp_f
    })
    results.append({
        '分类': category,
        '方法': 'Prophet', 'R2': r2_p, '残差STD': std_p, '幅度': amp_p
    })
    results.append({
        '分类': category,
        '方法': 'STL', 'R2': r2_s, '残差STD': std_s, '幅度': amp_s
    })

# ========== 汇总结果 ==========
result_df = pd.DataFrame(results)
result_df.sort_values(['分类', 'R2'], ascending=[True, False], inplace=True)
result_df.to_excel("拟合效果对比表.xlsx", index=False)
print("\n✅ 拟合完成，结果保存为：拟合效果对比表.xlsx")