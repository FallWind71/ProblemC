import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df_product = pd.read_excel(r"C:\Users\24621\Desktop\数模23c\附件1(2).xlsx")
df_sales = pd.read_excel(r"C:\Users\24621\Desktop\数模23c\附件2(1).xlsx")

# 打印列名以调试
print("销售数据列名:", df_sales.columns.tolist())
print("商品数据列名:", df_product.columns.tolist())

# 根据实际列名调整（常见问题：列名中的括号可能是全角/半角差异）
# 尝试不同的列名组合
possible_columns = {
    '销量': ['销量（千克）', '销量(千克)', '销量', '销量/kg', '销量(kg)'],
    '单价': ['销售单价（元/千克）', '销售单价(元/千克)', '单价', '价格', '销售单价']
}


# 找到实际存在的列名
def find_valid_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    # 如果没有匹配，尝试忽略括号
    for col in df.columns:
        if "销量" in col or "销售" in col:
            return col
    return None


# 获取有效列名
sales_col = find_valid_column(df_sales, possible_columns['销量'])
price_col = find_valid_column(df_sales, possible_columns['单价'])

if not sales_col or not price_col:
    raise ValueError("无法确定销量或单价列名，请检查数据")

print(f"使用销量列: {sales_col}, 单价列: {price_col}")

# 合并数据
df_merged = pd.merge(df_sales, df_product, on='单品编码', how='left')

# 数据预处理
df_merged['销售日期'] = pd.to_datetime(df_merged['销售日期'])
df_merged['销售额'] = df_merged[sales_col] * df_merged[price_col]

# 1. 品类销售分析
category_sales = df_merged.groupby('分类名称').agg(
    总销量_千克=(sales_col, 'sum'),
    总销售额=('销售额', 'sum'),
    单品数量=('单品名称', 'nunique')
).reset_index()

# 2. 单品销售分析
product_sales = df_merged.groupby(['分类名称', '单品名称']).agg(
    销量_千克=(sales_col, 'sum'),
    销售额=('销售额', 'sum'),
    销售天数=('销售日期', 'nunique')
).reset_index().sort_values('销量_千克', ascending=False)

# 3. 时间序列分析（按月汇总）
df_merged['月份'] = df_merged['销售日期'].dt.to_period('M')
monthly_sales = df_merged.groupby(['月份', '分类名称'])[sales_col].sum().unstack()

# 4. 单品关联性分析
# 创建每日单品销售矩阵
daily_sales = df_merged.groupby(['销售日期', '单品名称'])[sales_col].sum().unstack(fill_value=0)

# 计算单品间相关系数
correlation_matrix = daily_sales.corr()

# 筛选高销量单品
top_products = product_sales.head(10)['单品名称'].tolist()
top_corr_matrix = correlation_matrix.loc[top_products, top_products]

# 可视化
# 1. 品类销售分布
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.barplot(x='总销量_千克', y='分类名称',
            data=category_sales.sort_values('总销量_千克', ascending=False))
plt.title('各品类总销量分布')
plt.xlabel('总销量(千克)')
plt.ylabel('')

plt.subplot(2, 2, 2)
plt.pie(category_sales['总销量_千克'],
        labels=category_sales['分类名称'],
        autopct=lambda p: f'{p:.1f}%\n({p * sum(category_sales["总销量_千克"]) / 100:.0f}kg)',
        startangle=90)
plt.title('品类销量占比')

# 2. 单品销量分布
plt.subplot(2, 2, 3)
top_10_products = product_sales.head(10)
sns.barplot(x='销量_千克', y='单品名称',
            data=top_10_products.sort_values('销量_千克', ascending=True))
plt.title('销量Top10单品')
plt.xlabel('销量(千克)')
plt.ylabel('')

# 3. 时间序列分析
plt.subplot(2, 2, 4)
for category in monthly_sales.columns:
    plt.plot(monthly_sales.index.astype(str), monthly_sales[category],
             label=category, marker='o')
plt.title('各品类月度销售趋势')
plt.xlabel('月份')
plt.ylabel('销量(千克)')
plt.xticks(rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig('品类与单品分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 单品关联性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm',
            fmt=".2f", annot_kws={"size": 10}, linewidths=.5)
plt.title('高销量单品间销售相关性')
plt.tight_layout()
plt.savefig('单品关联性分析.png', dpi=300)
plt.show()

# 5. 各品类内单品分布
plt.figure(figsize=(14, 8))
for i, category in enumerate(category_sales['分类名称']):
    plt.subplot(2, 3, i + 1)
    category_data = product_sales[product_sales['分类名称'] == category].sort_values('销量_千克', ascending=False).head(5)
    sns.barplot(x='销量_千克', y='单品名称', data=category_data)
    plt.title(f'{category} Top5单品')
    plt.xlabel('销量(千克)')
    plt.ylabel('')

plt.tight_layout()
plt.savefig('各品类单品分布.png', dpi=300)
plt.show()

# 6. 额外分析：销售额与销量关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='销量_千克', y='销售额', hue='分类名称',
                size='销售天数', sizes=(20, 200),
                alpha=0.7, data=product_sales)
plt.title('单品销量与销售额关系')
plt.xlabel('总销量(千克)')
plt.ylabel('总销售额(元)')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('销量销售额关系.png', dpi=300)
plt.show()

# 输出关键分析结果
print("\n=== 关键分析结果 ===")
print(f"1. 品类销量排名: {category_sales.sort_values('总销量_千克', ascending=False)['分类名称'].tolist()}")
print(f"2. 最畅销单品: {top_10_products['单品名称'].head(3).tolist()}")
print(f"3. 最强正相关: {top_corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(3).to_dict()}")
print(f"4. 最强负相关: {top_corr_matrix.unstack().sort_values().drop_duplicates().head(3).to_dict()}")