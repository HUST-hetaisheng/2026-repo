import pandas as pd
import numpy as np
import os

# 读取数据
data_dir = r'd:\2026-repo\data'
consistency_file = os.path.join(data_dir, 'consistency_by_week_final.csv')
fan_vote_file = os.path.join(data_dir, 'fan_vote_results_final.csv')

df_cons = pd.read_csv(consistency_file)
df_vote = pd.read_csv(fan_vote_file)

# ==========================================
# 1. 按周计算的一致性几率 (Consistency by Week)
# ==========================================
# 聚合逻辑：按 Regime 和 Week 分组，计算 consistent 的平均值 (即概率)
# 为了展示方便，我们将生成一个透视表：行是 Week，列是 Regime

# 首先统一 Regime 名称大小写（如果有必要）
df_cons['regime'] = df_cons['regime'].str.lower()

# 计算分 Regime 的每周平均一致性
pivot_cons_week = df_cons.pivot_table(
    index='week', 
    columns='regime', 
    values='consistent', 
    aggfunc='mean'
)
# 计算整体的每周平均一致性
overall_cons_week = df_cons.groupby('week')['consistent'].mean()
pivot_cons_week['overall'] = overall_cons_week

# 保存
table1_path = os.path.join(data_dir, 'table_consistency_by_week.csv')
pivot_cons_week.to_csv(table1_path)
print(f"Generated: {table1_path}")

# ==========================================
# 2. 按季计算的一致性几率 (Consistency by Season)
# ==========================================
# 聚合逻辑：按 Season 计算 consistent 的平均值
# 同时保留 Regime 信息以便分析

season_cons = df_cons.groupby(['season', 'regime'])['consistent'].mean().reset_index()
season_cons.rename(columns={'consistent': 'consistency_prob'}, inplace=True)

# 保存
table2_path = os.path.join(data_dir, 'table_consistency_by_season.csv')
season_cons.to_csv(table2_path, index=False)
print(f"Generated: {table2_path}")

# ==========================================
# 3. 按周的不确定性 (Uncertainty by Week)
# ==========================================
# 不确定性指标：使用 CV (Coefficient of Variation)
# df_vote 中有 cv 列。我们需要按 Season/Week 聚合得到该周的平均 CV，然后再按 Week/Regime 聚合
# 或者直接按 Regime/Week 对所有选手的 CV 求平均

# 考虑到不同赛季选手数量不同，我们先计算每周所有选手的平均 CV，代表该周的整体不确定性
# 注意：fan_vote_results_final.csv 每一行是一个选手在某周的数据

df_vote['regime'] = df_vote['regime'].str.lower()

# 聚合逻辑：按 Regime 和 Week 分组，计算 CV 的平均值
pivot_uncert_week = df_vote.pivot_table(
    index='week', 
    columns='regime', 
    values='cv', 
    aggfunc='mean'
)

# 计算整体
overall_uncert_week = df_vote.groupby('week')['cv'].mean()
pivot_uncert_week['overall'] = overall_uncert_week

# 保存
table3_path = os.path.join(data_dir, 'table_uncertainty_by_week.csv')
pivot_uncert_week.to_csv(table3_path)
print(f"Generated: {table3_path}")

# ==========================================
# 4. 按季的不确定性 (Uncertainty by Season)
# ==========================================
# 聚合逻辑：按 Season 计算所有选手所有周的 CV 平均值

season_uncert = df_vote.groupby(['season', 'regime'])['cv'].mean().reset_index()
season_uncert.rename(columns={'cv': 'uncertainty_avg_cv'}, inplace=True)

# 保存
table4_path = os.path.join(data_dir, 'table_uncertainty_by_season.csv')
season_uncert.to_csv(table4_path, index=False)
print(f"Generated: {table4_path}")

# 打印预览
print("\n--- Preview: Consistency by Week ---")
print(pivot_cons_week.head())
print("\n--- Preview: Uncertainty by Week ---")
print(pivot_uncert_week.head())
