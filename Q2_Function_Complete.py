import os
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font="WenQuanYi Zen Hei", font_scale=1.2)

# 创建文件夹保存图像
os.makedirs("每品类成本预测", exist_ok=True)
df = pd.read_excel("每日品类加权成本.xlsx")
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 各品类需求函数定义
demand_functions = {
    "水生根茎类": {
        "func_type": "线性",
        "params": (-3.3695, 66.5862),
        "price_range": (4.39, 16.90)
    },
    "花叶类": {
        "func_type": "线性",
        "params": (-17.6906, 273.7243),
        "price_range": (2.54, 9.88)
    },
    "花菜类": {
        "func_type": "线性",
        "params": (-1.4875, 51.2450),
        "price_range": (4.32, 14.29)
    },
    "茄类": {
        "func_type": "指数",
        "params": (28.0734, -0.0389),
        "price_range": (3.00, 15.09)
    },
    "辣椒类": {
        "func_type": "反比例",
        "params": (265.2713, 41.7862),
        "price_range": (3.39, 16.66)
    },
    "食用菌": {
        "func_type": "线性",
        "params": (-3.2958, 89.9171),
        "price_range": (3.82, 15.69)
    }
}

# 各品类历史最大销量
max_sales = {
    "水生根茎类": 53.49 * 1.2,  # 64.19
    "花叶类": 233.37 * 1.2,    # 280.04
    "花菜类": 47.34 * 1.2,     # 56.81
    "茄类": 32.25 * 1.2,       # 38.70
    "辣椒类": 122.54 * 1.2,    # 147.05
    "食用菌": 79.28 * 1.2       # 95.14 (使用历史最大值)
}

# 需求函数计算
def calculate_demand(x, func_type, params):
    if func_type == "线性":
        a, b = params
        return a * x + b
    elif func_type == "指数":
        a, b = params
        return a * np.exp(b * x)
    elif func_type == "反比例":
        a, b = params
        return a / (x + 1e-5) + b  # 避免除以0
    else:
        raise ValueError(f"未知函数类型: {func_type}")

# 带约束的需求函数
def bounded_demand(x, func_type, params, category):
    raw = calculate_demand(x, func_type, params)
    return max(0, min(raw, max_sales[category]))

# 定价优化函数
def optimize_price(cost, func_type, params, category, price_range):
    best_price, best_profit = None, -float('inf')
    best_sales = 0
    
    # 在价格范围内采样100个点
    prices = np.linspace(price_range[0], price_range[1], 100)
    for p in prices:
        # 跳过低于成本的价格
        if p < cost:
            continue
            
        sales = bounded_demand(p, func_type, params, category)
        profit = (p - cost) * sales
        
        # 寻找最大收益点
        if profit > best_profit:
            best_profit = profit
            best_price = p
            best_sales = sales
    
    # 如果没有找到可行解
    if best_price is None:
        return price_range[1], 0, 0
    
    return best_price, best_sales, best_profit

# 预测未来一周成本并优化定价
results = []

for category in df['分类名称'].unique():
    # 准备Prophet数据
    category_df = df[df['分类名称'] == category][['销售日期', '加权单位成本']]
    category_df.columns = ['ds', 'y']
    
    # 训练Prophet模型
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    model.add_country_holidays(country_name='CN')
    model.fit(category_df)
    
    # 创建未来日期（2023-07-01 到 2023-07-07）
    future = pd.DataFrame({
        'ds': pd.date_range(start='2023-07-01', periods=7)
    })
    
    # 预测成本
    forecast = model.predict(future)
    forecast['category'] = category
    
    # 可视化预测结果
    fig = model.plot(forecast)
    plt.title(f'{category}成本预测')
    plt.xlabel('日期')
    plt.ylabel('加权单位成本（元/千克）')
    plt.tight_layout()
    plt.savefig(f"每品类成本预测/{category}_成本预测.png")
    plt.close()
    
    # 优化每日定价和补货量
    func_info = demand_functions[category]
    func_type = func_info["func_type"]
    params = func_info["params"]
    price_range = func_info["price_range"]
    
    for i, row in forecast.iterrows():
        date = row['ds']
        cost = row['yhat']
        
        # 优化定价
        optimal_price, optimal_sales, profit = optimize_price(
            cost, func_type, params, category, price_range
        )
        
        results.append({
            '日期': date.strftime('%Y-%m-%d'),
            '品类': category,
            '预测成本(元/千克)': round(cost, 2),
            '最优定价(元/千克)': round(optimal_price, 2),
            '补货总量(千克)': round(optimal_sales, 2),
            '预期收益(元)': round(profit, 2)
        })

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_excel("问题二_各品类补货与定价策略_新版.xlsx", index=False)

# 打印最终结果
print("\n🎯 问题二最终策略（新版）：")
print(results_df[['日期', '品类', '预测成本(元/千克)', '最优定价(元/千克)', '补货总量(千克)', '预期收益(元)']])

# 可视化各品类定价策略
plt.figure(figsize=(12, 8))
for category in results_df['品类'].unique():
    cat_df = results_df[results_df['品类'] == category]
    plt.plot(cat_df['日期'], cat_df['最优定价(元/千克)'], 'o-', label=category)
    
plt.title('各品类未来一周最优定价策略')
plt.xlabel('日期')
plt.ylabel('最优定价(元/千克)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("各品类最优定价策略_新版.png")
plt.close()

print("\n✅ 新版结果已保存到'问题二_各品类补货与定价策略_新版.xlsx'")