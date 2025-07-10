import pandas as pd
import numpy as np

##################
# 读取数据
##################
df1 = pd.read_excel('附件1.xlsx')  # 商品信息（含单品编码、品类）
print("附件一读取完成\n")

df2 = pd.read_excel('附件2.xlsx')  # 销售记录（含销量、价格等）
print("附件二读取完成\n")

df3 = pd.read_excel('附件3.xlsx')  # 批发价记录（按日期+编码）
print("附件三读取完成\n")

##################
# 合并数据
##################

# 第一步：销售数据 + 单品分类信息（on 单品编码）
#merge的效果，简单来说，就是按照设定的值“单品编码“完成两个表格的一一配对，把表格一的对应值加到表格二的对应行
df_1 = df2.merge(df1, on="单品编码", how="left")
print("附件1, 附件2合并完成\n")

# 第二步：再和批发价格合并（on 单品编码 + 日期）
df3.rename(columns={"日期": "销售日期"}, inplace=True)  # 保证列名一致
df = df_1.merge(df3, on=["销售日期", "单品编码"], how="left")
print("附件1,2,3合并完成\n")

# 检查是否有缺失
print("缺失值统计如下：")
print(df.isnull().sum())

##################
# 日期处理
##################
#识别日期，转化为标准格式，方便后续对季度，月，年的分割
df['销售日期'] = pd.to_datetime(df['销售日期'])
print("日期格式转换完成\n")

#对所有数据分别加上该数据属于哪个年，月，季度的标记
df['年'] = df['销售日期'].dt.year
df['季度'] = df['销售日期'].dt.quarter
df['月'] = df['销售日期'].dt.month
df['日'] = df['销售日期'].dt.day

print("添加 年、季度、月、日 完成\n")

#调试用，用于输出示例数据
print("示例日期如下：")
print(df[['销售日期', '年', '季度', '月','日']].head())

##################
# 计算销售额与成本
##################
#添加销售额和成本列
df['销售额'] = df['销量(千克)'] * df['销售单价(元/千克)']
df['成本'] = df['销量(千克)'] * df['批发价格(元/千克)']
print("销售额与成本计算完成\n")

##################
# 统计：季度
##################
#分出以每年每季度每单品的组，对组内'销量(千克)', '销售额'分别求和，然后转化为表格列
quarterly_item = df.groupby(['年', '季度', '单品名称'])[['销量(千克)', '销售额']].sum().reset_index()
#输出表格文件
quarterly_item.to_excel("单品季度销量.xlsx", index=False)
print("单品季度销量.xlsx 输出完成\n")

#分出以每年每季度每分类的组，对组内'销量(千克)', '销售额'分别求和，然后转化为表格列
quarterly_class = df.groupby(['年', '季度', '分类名称'])[['销量(千克)', '销售额']].sum().reset_index()
#输出表格文件
quarterly_class.to_excel("品类季度销量.xlsx", index=False)
print("品类季度销量.xlsx 输出完成\n")

##################
# 统计：月度
##################
#分出以每年每季度每分类的组，对组内'销量(千克)', '销售额'，‘成本’分别求和，然后转化为表格列
monthly_item = df.groupby(['年', '月', '分类名称'])[['销量(千克)', '销售额', '成本']].sum().reset_index()
monthly_item.to_excel("品类销售额_销量_成本月度统计.xlsx", index=False)
print("月度 品类销售额_销量_成本.xlsx 输出完成\n")

##################
# 统计：日
##################
# 新增列：单位成本 = 成本 / 销量
df['单位成本'] = df['成本'] / df['销量(千克)']
df['单位售价'] = df['销售单价(元/千克)']


# 去除无效值（如销量为 0）
df = df[df['销量(千克)'] > 0]

# 计算每天每品类的总销量
total_sales = df.groupby(['销售日期', '分类名称'])['销量(千克)'].transform('sum')

# 每条记录销量在当天该分类中的占比（权重）
df['销量占比'] = df['销量(千克)'] / total_sales

# 加权单位成本 = 单品单位成本 × 当日销量占比
df['加权单位成本'] = df['单位成本'] * df['销量占比']
df['加权单位售价'] = df['单位售价'] * df['销量占比']


# 按天、品类聚合，加权单位成本求和 × 总销量 => 得到该品类每天总成本
daily_category_cost = df.groupby(['销售日期', '分类名称']).apply(
    lambda g: pd.Series({
        '加权单位成本': g['加权单位成本'].sum(),
        '加权单位售价': g['加权单位售价'].sum(),  # ⬅ 新增
        '总销量': g['销量(千克)'].sum(),
        '总成本': g['加权单位成本'].sum() * g['销量(千克)'].sum(),
        '总销售额': g['加权单位售价'].sum() * g['销量(千克)'].sum()  # ⬅ 可选
    }),

).reset_index()

# 拆分日期
daily_category_cost['年'] = daily_category_cost['销售日期'].dt.year
daily_category_cost['月'] = daily_category_cost['销售日期'].dt.month
daily_category_cost['日'] = daily_category_cost['销售日期'].dt.day

# 保存
daily_category_cost.to_excel("每日品类加权成本.xlsx", index=False)
print("每日品类加权成本.xlsx 输出完成\n")
