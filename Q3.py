
import pandas as pd
import numpy as np
import math

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 读取数据
print("读取数据...")
df1 = pd.read_excel('附件1.xlsx')  # 单品到分类映射
df2 = pd.read_excel('附件2.xlsx')  # 销售数据
df3 = pd.read_excel('附件3.xlsx')  # 批发价格数据
df4 = pd.read_excel('附件4.xlsx')  # 损耗率数据
df1.info()
df2.info()
df3.info()
df4.info()

# 数据预处理
print("处理销售数据...")
# 提取2023年6月24-30日的可售单品
available_items = df2[(df2['销售日期'] >= '2023-06-24') & 
                      (df2['销售日期'] <= '2023-06-30')]['单品编码'].unique()

# 聚合销售数据，按单品和日期计算总销售量和平均销售单价
daily_sales = df2.groupby(['单品编码', '销售日期']).agg({
    '销量(千克)': 'sum',
    '销售单价(元/千克)': 'mean'
}).reset_index()

# 合并批发价格
daily_sales = daily_sales.merge(df3[['单品编码', '批发价格(元/千克)']], on='单品编码', how='left')

# 映射损耗率
item_category_map = df1[['单品编码', '分类编码', '分类名称']]
df4 = df4.rename(columns={'小分类编码': '分类编码', '平均损耗率(%)_小分类编码_不同值': '损耗率(%)'})
category_loss = df4[['分类编码', '损耗率(%)']]
item_loss = item_category_map.merge(category_loss, on='分类编码', how='left')
daily_sales = daily_sales.merge(item_loss[['单品编码', '损耗率(%)']], on='单品编码', how='left')

# 定义品类需求函数参数
category_demand_functions = {
    '水生根茎类': {'type': 'linear', 'a': 66.5862, 'b': 3.3695},
    '花叶类': {'type': 'linear', 'a': 273.7243, 'b': 17.6906},
    '花菜类': {'type': 'linear', 'a': 51.2450, 'b': 1.4875},
    '茄类': {'type': 'exponential', 'a': 28.0734, 'b': 0.0389},
    '辣椒类': {'type': 'reciprocal', 'a': 265.2713, 'b': 41.7862},
    '食用菌': {'type': 'linear', 'a': 89.9171, 'b': 3.2958}
}

# 获取单品的需求函数参数
def get_demand_function(item_code):
    """根据单品编码获取所属品类的需求函数参数"""
    category = item_loss[item_loss['单品编码'] == item_code]['分类名称'].values[0]
    return category_demand_functions.get(category, {'type': 'linear', 'a': 10, 'b': 0.5})  # 默认参数

# 计算销售量
def calculate_sales(p, demand_params):
    """根据定价和需求函数参数计算销售量"""
    func_type = demand_params['type']
    if func_type == 'linear':
        a, b = demand_params['a'], demand_params['b']
        s = a - b * p
    elif func_type == 'exponential':
        a, b = demand_params['a'], demand_params['b']
        s = a * math.exp(-b * p)
    elif func_type == 'reciprocal':
        a, b = demand_params['a'], demand_params['b']
        s = a / p + b
    return max(s, 0)  # 确保销售量非负

# 计算收益
def calculate_profit(p, demand_params, w, d):
    """计算单品收益"""
    s = calculate_sales(p, demand_params)  # 预测销售量
    if s <= 0:
        return -np.inf  # 无效定价
    d = d / 100  # 损耗率从百分比转换为小数
    q = s / (1 - d)  # 补货量考虑损耗
    q = max(q, 2.5)  # 满足最小陈列量
    profit = p * s - w * q
    return profit

# 寻找最优定价
def find_optimal_p(item_code, demand_params, w, d):
    """通过网格搜索找到最优定价"""
    p_values = np.arange(w, w + 10, 0.1)  # 定价范围：批发价到批发价+10
    profits = [calculate_profit(p, demand_params, w, d) for p in p_values]
    max_profit_idx = np.argmax(profits)
    optimal_p = p_values[max_profit_idx]
    optimal_profit = profits[max_profit_idx]
    return optimal_p, optimal_profit

# 计算每个单品的最优定价和收益
print("优化定价并计算收益...")
item_info = df3.merge(item_loss, on='单品编码')
profits = {}
for item in available_items:
    demand_params = get_demand_function(item)
    w = item_info[item_info['单品编码'] == item]['批发价格(元/千克)'].values[0]
    d = item_info[item_info['单品编码'] == item]['损耗率(%)'].values[0]
    optimal_p, optimal_profit = find_optimal_p(item, demand_params, w, d)
    profits[item] = (optimal_p, optimal_profit)

# 按收益降序排序并选择单品
print("选择单品...")
sorted_items = sorted(profits.items(), key=lambda x: x[1][1], reverse=True)
num_items = min(max(27, len(available_items)), 33)  # 选择27-33个单品
selected_items = sorted_items[:num_items]

# 输出结果
print(f"\n7月1日单品补货量和定价策略（共{len(selected_items)}个单品）：")
print("单品编码 | 补货量 (千克) | 定价 (元/千克) | 预计收益 (元)")
total_profit = 0
for item, (p, profit) in selected_items:
    demand_params = get_demand_function(item)
    s = calculate_sales(p, demand_params)
    d = item_info[item_info['单品编码'] == item]['损耗率(%)'].values[0] / 100
    q = max(2.5, s / (1 - d))
    print(f"{item} | {q:.2f} | {p:.2f} | {profit:.2f}")
    total_profit += profit

print(f"\n总预计收益: {total_profit:.2f} 元")
