# -*- coding: utf-8 -*-
"""分析 λ=0.02 赛季的特征"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'd:\2026-repo\data\task4_optimal_results_v3.csv')

# 分离 lambda=0.02 和其他
low_lambda = df[df['optimal_lambda'] == 0.02]
other = df[df['optimal_lambda'] != 0.02]

print('='*60)
print('λ=0.02 的赛季分析')
print('='*60)
print(f'数量: {len(low_lambda)} 个赛季')
print(f'赛季: {low_lambda["season"].tolist()}')
print()

# 对比两组的指标
print('原始指标对比 (λ=0.02 vs 其他):')
print('-'*60)
for col in ['celebrity_raw', 'dancer_raw', 'ccvi_raw', 'satisfaction_raw', 'discrepancy_raw']:
    mean_low = low_lambda[col].mean()
    mean_other = other[col].mean()
    print(f'{col:25s}: {mean_low:10.2f} vs {mean_other:10.2f}  (差异: {mean_low-mean_other:+.2f})')

print()
print('归一化指标对比:')
print('-'*60)
for col in ['celebrity_norm', 'dancer_norm', 'ccvi_norm', 'satisfaction_norm', 'discrepancy_benefit']:
    mean_low = low_lambda[col].mean()
    mean_other = other[col].mean()
    print(f'{col:25s}: {mean_low:.3f} vs {mean_other:.3f}  (差异: {mean_low-mean_other:+.3f})')

print()
print('='*60)
print('关键发现: discrepancy_raw 详细分析')
print('='*60)
print('λ=0.02 赛季的 discrepancy_raw:')
for _, row in low_lambda.iterrows():
    print(f"  Season {row['season']:2.0f}: discrepancy={row['discrepancy_raw']:.4f}")
print(f'\n  平均: {low_lambda["discrepancy_raw"].mean():.3f}')
print(f'  其他赛季平均: {other["discrepancy_raw"].mean():.3f}')

print()
print('='*60)
print('问题诊断: 为什么 λ=0.02 会"最优"?')
print('='*60)

# 检查 discrepancy_benefit 列
print('\ndiscrepancy_benefit 分析（越高越好，因为 discrepancy 是负面指标）:')
print('-'*60)
print(f'λ=0.02 赛季 discrepancy_benefit 平均: {low_lambda["discrepancy_benefit"].mean():.3f}')
print(f'其他赛季 discrepancy_benefit 平均: {other["discrepancy_benefit"].mean():.3f}')

# 检查 satisfaction
print(f'\nλ=0.02 赛季 satisfaction_norm 平均: {low_lambda["satisfaction_norm"].mean():.3f}')
print(f'其他赛季 satisfaction_norm 平均: {other["satisfaction_norm"].mean():.3f}')

# 分析收益公式
print('\n收益公式分解 (w1=0.2, w2=0.1, w3=0.3, w4=0.2, w5=0.2):')
print('-'*60)
w = [0.2, 0.1, 0.3, 0.2, 0.2]
cols = ['celebrity_norm', 'dancer_norm', 'ccvi_norm', 'satisfaction_norm', 'discrepancy_benefit']
names = ['Celebrity', 'Dancer', 'CCVI', 'Satisfaction', 'Discrepancy']

for i, (col, name, wi) in enumerate(zip(cols, names, w)):
    contrib_low = low_lambda[col].mean() * wi
    contrib_other = other[col].mean() * wi
    print(f'{name:15s} (w={wi}): {contrib_low:.3f} vs {contrib_other:.3f}  (差异: {contrib_low-contrib_other:+.3f})')

print()
print('='*60)
print('根本原因分析')
print('='*60)

# 看一下 Season 13 特别分析（discrepancy 接近 0）
s13 = df[df['season'] == 13]
print('\nSeason 13 详细分析（discrepancy_raw ≈ 0）:')
print(s13[['season', 'optimal_lambda', 'discrepancy_raw', 'discrepancy_benefit', 'satisfaction_raw']].to_string())

# 看一下 Season 26
s26 = df[df['season'] == 26]
print('\nSeason 26 详细分析:')
print(s26[['season', 'optimal_lambda', 'discrepancy_raw', 'discrepancy_benefit', 'celebrity_raw', 'ccvi_raw']].to_string())

print('\n' + '='*60)
print('结论')
print('='*60)
print("""
问题根源：
1. discrepancy_raw 接近 0 → discrepancy_benefit = 1.0（最高）
2. 当 λ 很小时，模拟结果接近原始粉丝投票，discrepancy 自然很小
3. 这形成了"自我强化"：λ↓ → 模拟更接近粉丝 → discrepancy↓ → benefit↑

这是一个逻辑缺陷：
- Discrepancy 应该衡量"评委与粉丝的分歧"
- 但现在衡量的是"模拟结果与原始粉丝的差异"
- 当 λ→0，模拟完全由粉丝决定，discrepancy 自然趋近 0
""")
