"""
绘制 Rank vs Percentage 方法对观众投票偏向的对比图

核心发现：Percentage 方法更偏向于保护观众喜爱的选手
- Pct 偏向观众次数: 56 vs Rank 偏向观众次数: 37
- Pct 准确率: 69.5% vs Rank 准确率: 61.1%
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 路径设置
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 读取预计算的对比数据
df = pd.read_csv(DATA_DIR / "rank_vs_pct_comparison.csv")

# ---------- 统计量计算 ----------
# 两种方法不一致的周数
disagree = df[df['methods_agree'] == False]
n_disagree = len(disagree)

# 各方法偏向观众的次数
fan_favored_by_pct = (disagree['fan_favored_by'] == 'Percentage').sum()
fan_favored_by_rank = (disagree['fan_favored_by'] == 'Rank').sum()

# 准确率（预测实际淘汰者）
rank_correct = (df['rank_eliminated'] == df['actual_eliminated']).sum()
pct_correct = (df['pct_eliminated'] == df['actual_eliminated']).sum()
n_total = len(df)
rank_acc = rank_correct / n_total * 100
pct_acc = pct_correct / n_total * 100

print(f"两方法不一致周数: {n_disagree}")
print(f"Pct偏向观众: {fan_favored_by_pct}, Rank偏向观众: {fan_favored_by_rank}")
print(f"Rank准确率: {rank_acc:.1f}%, Pct准确率: {pct_acc:.1f}%")

# ---------- 图1: 方法偏向性对比（条形图）----------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# 子图1: 偏向观众的次数
ax1 = axes[0]
methods = ['Percentage\nMethod', 'Rank\nMethod']
fan_favored_counts = [fan_favored_by_pct, fan_favored_by_rank]
colors = ['#2ecc71', '#e74c3c']  # 绿色=Pct更好, 红色=Rank
bars1 = ax1.bar(methods, fan_favored_counts, color=colors, edgecolor='black', linewidth=1.2)

# 添加数值标签
for bar, count in zip(bars1, fan_favored_counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_ylabel('Number of Weeks', fontsize=12)
ax1.set_title('Fan-Favored Outcomes\n(When Methods Disagree)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(fan_favored_counts) * 1.2)

# 添加比例标注
ratio = fan_favored_by_pct / fan_favored_by_rank if fan_favored_by_rank > 0 else float('inf')
ax1.text(0.5, 0.92, f'Ratio: {ratio:.2f}×', transform=ax1.transAxes, 
         ha='center', fontsize=11, color='#2c3e50',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

# 子图2: 预测准确率
ax2 = axes[1]
accuracies = [pct_acc, rank_acc]
bars2 = ax2.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2)

for bar, acc in zip(bars2, accuracies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Prediction Accuracy\n(vs Actual Elimination)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 100)

# 添加差异标注
diff = pct_acc - rank_acc
ax2.text(0.5, 0.92, f'Δ = +{diff:.1f}%', transform=ax2.transAxes, 
         ha='center', fontsize=11, color='#27ae60',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

# 子图3: delta_fan 分布（Pct淘汰者 vs Rank淘汰者的观众支持率差）
ax3 = axes[2]
delta_fan = disagree['delta_fan'].dropna()

# delta_fan > 0 表示 Pct 淘汰者的观众支持率更高（即 Pct 保护了观众喜爱的选手）
# delta_fan < 0 表示 Rank 淘汰者的观众支持率更高

# 根据正负着色
pos_delta = delta_fan[delta_fan > 0]
neg_delta = delta_fan[delta_fan <= 0]

bins = np.linspace(delta_fan.min() - 0.5, delta_fan.max() + 0.5, 20)
ax3.hist(pos_delta, bins=bins, color='#2ecc71', alpha=0.7, label='Pct Favors Fan (Δ>0)', edgecolor='black')
ax3.hist(neg_delta, bins=bins, color='#e74c3c', alpha=0.7, label='Rank Favors Fan (Δ≤0)', edgecolor='black')

ax3.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax3.set_xlabel('Δ Fan Vote (Pct_elim − Rank_elim)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Fan Vote Difference Distribution\n(Disagreement Weeks)', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)

# 添加均值标注
mean_delta = delta_fan.mean()
ax3.axvline(mean_delta, color='#3498db', linestyle='-', linewidth=2, alpha=0.8)
ax3.text(mean_delta, ax3.get_ylim()[1] * 0.9, f'Mean={mean_delta:.2f}', 
         fontsize=10, color='#3498db', ha='left' if mean_delta > 0 else 'right')

plt.tight_layout()
plt.savefig(FIG_DIR / "rank_vs_pct_fan_bias.png", dpi=800, bbox_inches='tight')
plt.savefig(FIG_DIR / "rank_vs_pct_fan_bias.pdf", bbox_inches='tight')
plt.close()

print(f"\n图像已保存: {FIG_DIR / 'rank_vs_pct_fan_bias.png'}")

# ---------- 图2: 按 regime 分组的偏向性分析 ----------
fig2, ax = plt.subplots(figsize=(10, 6))

# 按 regime 分组统计
regime_stats = []
for regime in df['regime'].unique():
    regime_df = df[df['regime'] == regime]
    disagree_regime = regime_df[regime_df['methods_agree'] == False]
    
    pct_favor = (disagree_regime['fan_favored_by'] == 'Percentage').sum()
    rank_favor = (disagree_regime['fan_favored_by'] == 'Rank').sum()
    total_disagree = len(disagree_regime)
    
    if total_disagree > 0:
        pct_ratio = pct_favor / total_disagree * 100
        rank_ratio = rank_favor / total_disagree * 100
    else:
        pct_ratio = rank_ratio = 0
    
    regime_stats.append({
        'regime': regime,
        'pct_favor': pct_favor,
        'rank_favor': rank_favor,
        'pct_ratio': pct_ratio,
        'total_disagree': total_disagree
    })

regime_df_stats = pd.DataFrame(regime_stats)
regime_df_stats = regime_df_stats.sort_values('regime')

# 堆叠条形图
x = np.arange(len(regime_df_stats))
width = 0.6

bars_pct = ax.bar(x, regime_df_stats['pct_favor'], width, label='Pct Favors Fan', color='#2ecc71', edgecolor='black')
bars_rank = ax.bar(x, regime_df_stats['rank_favor'], width, bottom=regime_df_stats['pct_favor'], 
                   label='Rank Favors Fan', color='#e74c3c', edgecolor='black')

# 添加比例标签
for i, (idx, row) in enumerate(regime_df_stats.iterrows()):
    total = row['pct_favor'] + row['rank_favor']
    if total > 0:
        ax.text(i, total + 0.5, f"{row['pct_ratio']:.0f}%\n({int(row['pct_favor'])}/{int(total)})", 
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Competition Regime', fontsize=12)
ax.set_ylabel('Number of Disagreement Weeks', fontsize=12)
ax.set_title('Fan-Favored Outcomes by Regime\n(Percentage vs Rank Method)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regime_df_stats['regime'], rotation=0)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIG_DIR / "rank_vs_pct_fan_bias_by_regime.png", dpi=800, bbox_inches='tight')
plt.savefig(FIG_DIR / "rank_vs_pct_fan_bias_by_regime.pdf", bbox_inches='tight')
plt.close()

print(f"图像已保存: {FIG_DIR / 'rank_vs_pct_fan_bias_by_regime.png'}")

# ---------- 输出 LaTeX 表格 ----------
print("\n" + "="*60)
print("LaTeX 表格输出")
print("="*60)

latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of Rank vs Percentage Method: Fan Vote Bias}
\label{tab:method_fan_bias}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Percentage} & \textbf{Rank} \\
\midrule
Fan-Favored Outcomes (count) & """ + str(fan_favored_by_pct) + r""" & """ + str(fan_favored_by_rank) + r""" \\
Fan-Favored Ratio (\%) & """ + f"{fan_favored_by_pct/n_disagree*100:.1f}" + r""" & """ + f"{fan_favored_by_rank/n_disagree*100:.1f}" + r""" \\
Prediction Accuracy (\%) & """ + f"{pct_acc:.1f}" + r""" & """ + f"{rank_acc:.1f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
print(latex_table)
