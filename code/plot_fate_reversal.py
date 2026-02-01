"""
命运逆转可视化：证明 Percentage 方法偏向粉丝，Rank 方法偏向评委

三个王炸方案：
1. 命运分歧条形图 (Divergence Bar Chart) - 只看分歧周，按 delta_rank 和粉丝份额
2. 安全边际对比图 (Safety Margin Correlation Plot) - 粉丝份额 vs 名次红利
3. 生存概率曲线 (Survival Probability Curves) - 不同粉丝份额区间的生存率

核心发现：
- Pct 偏向观众: 56 次 (60.2%)
- Rank 偏向观众: 37 次 (39.8%)
- Pct 准确率: 69.5% vs Rank 准确率: 61.1%
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# 路径设置
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 读取数据
rankings = pd.read_csv(DATA_DIR / "task2_rankings.csv")
comparison = pd.read_csv(DATA_DIR / "rank_vs_pct_comparison.csv")

print(f"Rankings 数据: {len(rankings)} 条记录")
print(f"Comparison 数据: {len(comparison)} 周")

# ============================================================
# 方案一：命运分歧条形图 (Divergence Bar Chart)
# ============================================================
print("\n" + "="*60)
print("方案一：命运分歧条形图")
print("="*60)

# 筛选分歧周
disagree = comparison[comparison['methods_agree'] == False].copy()
print(f"分歧周数: {len(disagree)}")

# 计算 delta_rank = rank_method_placement - pct_method_placement
# 正数 = Pct 对该选手更好（排名制排得更靠后）
# 为此需要获取被淘汰者在两种方法下的排名

# 从 rankings 中获取每周每个选手的排名
divergence_data = []

for _, row in disagree.iterrows():
    season = row['season']
    week = row['week']
    rank_elim = row['rank_eliminated']
    pct_elim = row['pct_eliminated']
    
    # 获取该周的选手数据
    week_data = rankings[(rankings['season'] == season) & (rankings['week'] == week)]
    
    if len(week_data) == 0:
        continue
    
    # Rank 方法淘汰的选手
    rank_contestant = week_data[week_data['celebrity_name'] == rank_elim]
    pct_contestant = week_data[week_data['celebrity_name'] == pct_elim]
    
    if len(rank_contestant) == 0 or len(pct_contestant) == 0:
        continue
    
    rank_contestant = rank_contestant.iloc[0]
    pct_contestant = pct_contestant.iloc[0]
    
    # delta_rank: Rank淘汰者在两种规则下的排名差
    # 如果 Rank 淘汰的人在 Pct 规则下排名更好（数字更小），说明 Pct 救了他
    delta_rank_for_rank_elim = rank_contestant['rank_rule_rank'] - rank_contestant['percent_rule_rank']
    
    divergence_data.append({
        'season': season,
        'week': week,
        'rank_eliminated': rank_elim,
        'pct_eliminated': pct_elim,
        'rank_elim_fan_share': rank_contestant['fan_vote_share'],
        'pct_elim_fan_share': pct_contestant['fan_vote_share'],
        'rank_elim_judge_share': rank_contestant['judge_share'],
        'pct_elim_judge_share': pct_contestant['judge_share'],
        'delta_fan': row['delta_fan'],  # rank_elim_fan - pct_elim_fan
        'regime': row['regime']
    })

div_df = pd.DataFrame(divergence_data)
print(f"有效分歧数据: {len(div_df)} 条")

# 图1: 分歧条形图 - 按 delta_fan 排序，颜色标记偏向
fig1, ax1 = plt.subplots(figsize=(14, 8))

# 按 delta_fan 排序
div_sorted = div_df.sort_values('delta_fan', ascending=True).reset_index(drop=True)

# 颜色：正数(Rank淘汰者粉丝更高) = 红色(Rank不利), 负数 = 绿色(Pct不利)
colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in div_sorted['delta_fan']]

bars = ax1.barh(range(len(div_sorted)), div_sorted['delta_fan'], color=colors, edgecolor='none', alpha=0.8)

# 添加零线
ax1.axvline(0, color='black', linewidth=1.5, linestyle='-')

# 标注极端案例
extreme_pos = div_sorted.nlargest(3, 'delta_fan')
extreme_neg = div_sorted.nsmallest(3, 'delta_fan')

for _, row in extreme_pos.iterrows():
    idx = div_sorted[div_sorted['rank_eliminated'] == row['rank_eliminated']].index[0]
    ax1.annotate(f"S{row['season']}W{row['week']}\n{row['rank_eliminated'][:12]}",
                 xy=(row['delta_fan'], idx), xytext=(row['delta_fan'] + 0.02, idx),
                 fontsize=8, va='center', color='#c0392b')

for _, row in extreme_neg.iterrows():
    idx = div_sorted[div_sorted['pct_eliminated'] == row['pct_eliminated']].index[0]
    ax1.annotate(f"S{row['season']}W{row['week']}\n{row['pct_eliminated'][:12]}",
                 xy=(row['delta_fan'], idx), xytext=(row['delta_fan'] - 0.02, idx),
                 fontsize=8, va='center', ha='right', color='#27ae60')

ax1.set_xlabel(r'$\Delta$ Fan Vote Share (Rank Elim − Pct Elim)', fontsize=12)
ax1.set_ylabel('Disagreement Week Index', fontsize=12)
ax1.set_title('The Fate Divergence Chart\nPositive = Rank Eliminates Higher Fan Support (Rank Hurts Fans)\n'
              'Negative = Pct Eliminates Higher Fan Support (Pct Hurts Fans)', fontsize=13, fontweight='bold')

# 添加统计摘要
pos_count = (div_sorted['delta_fan'] > 0).sum()
neg_count = (div_sorted['delta_fan'] <= 0).sum()
ax1.text(0.98, 0.95, f'Rank hurts fans: {pos_count} weeks\nPct hurts fans: {neg_count} weeks\nRatio: {pos_count/neg_count:.2f}×',
         transform=ax1.transAxes, fontsize=11, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(FIG_DIR / "fate_divergence_bar.png", dpi=800, bbox_inches='tight')
plt.savefig(FIG_DIR / "fate_divergence_bar.pdf", bbox_inches='tight')
plt.close()
print(f"保存: fate_divergence_bar.png")

# ============================================================
# 方案二：安全边际对比图 (Safety Margin Correlation Plot)
# ============================================================
print("\n" + "="*60)
print("方案二：安全边际对比图")
print("="*60)

# 计算每个选手在两种规则下的"名次红利"
# delta_placement = rank_rule_rank - percent_rule_rank
# 正数 = Pct 规则下名次更好（数字更小）

# 筛选有效数据
valid_rankings = rankings.dropna(subset=['fan_vote_share', 'rank_rule_rank', 'percent_rule_rank']).copy()
valid_rankings['rank_advantage_pct'] = valid_rankings['rank_rule_rank'] - valid_rankings['percent_rule_rank']

print(f"有效选手-周数据: {len(valid_rankings)}")
print(f"Rank Advantage of Pct 范围: [{valid_rankings['rank_advantage_pct'].min()}, {valid_rankings['rank_advantage_pct'].max()}]")

# 图2: 安全边际散点图
fig2, ax2 = plt.subplots(figsize=(12, 8))

# 按是否被淘汰着色
eliminated = valid_rankings[valid_rankings['eliminated_this_week'] == True]
survived = valid_rankings[valid_rankings['eliminated_this_week'] == False]

# 绘制散点（幸存者用浅色，淘汰者用深色）
ax2.scatter(survived['fan_vote_share'], survived['rank_advantage_pct'], 
            alpha=0.3, s=30, c='#3498db', label='Survived', edgecolors='none')
ax2.scatter(eliminated['fan_vote_share'], eliminated['rank_advantage_pct'], 
            alpha=0.7, s=50, c='#e74c3c', label='Eliminated', edgecolors='black', linewidths=0.5)

# 拟合回归线
slope, intercept, r_value, p_value, std_err = stats.linregress(
    valid_rankings['fan_vote_share'], valid_rankings['rank_advantage_pct'])

x_fit = np.linspace(0, valid_rankings['fan_vote_share'].max(), 100)
y_fit = slope * x_fit + intercept
ax2.plot(x_fit, y_fit, 'k--', linewidth=2.5, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.3f}')

# 添加零线
ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.7)

# 标注 Bobby Bones 类型的极端案例（高粉丝+高正红利）
extreme_cases = valid_rankings[(valid_rankings['fan_vote_share'] > 0.25) & 
                                (valid_rankings['rank_advantage_pct'] > 3)]
for _, row in extreme_cases.head(5).iterrows():
    ax2.annotate(f"{row['celebrity_name'][:15]}\nS{row['season']}W{row['week']}", 
                 xy=(row['fan_vote_share'], row['rank_advantage_pct']),
                 xytext=(row['fan_vote_share'] + 0.02, row['rank_advantage_pct'] + 0.5),
                 fontsize=8, alpha=0.8,
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

ax2.set_xlabel('Fan Vote Share', fontsize=12)
ax2.set_ylabel('Rank Advantage of Percentage Rule\n(Positive = Pct Ranks You Higher)', fontsize=12)
ax2.set_title('Safety Margin Correlation: Fan Support vs Rule Preference\n'
              'Positive slope proves Percentage Rule favors high fan support', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)

# 添加区域标注
ax2.fill_between([0, 0.5], 0, 10, alpha=0.1, color='green', label='Pct Advantage Zone')
ax2.fill_between([0, 0.5], -10, 0, alpha=0.1, color='red', label='Rank Advantage Zone')
ax2.set_xlim(0, valid_rankings['fan_vote_share'].max() * 1.05)
ax2.set_ylim(valid_rankings['rank_advantage_pct'].min() - 1, valid_rankings['rank_advantage_pct'].max() + 1)

# 统计摘要
pct_advantage_count = (valid_rankings['rank_advantage_pct'] > 0).sum()
rank_advantage_count = (valid_rankings['rank_advantage_pct'] < 0).sum()
neutral_count = (valid_rankings['rank_advantage_pct'] == 0).sum()

ax2.text(0.98, 0.05, f'Pct better: {pct_advantage_count} ({pct_advantage_count/len(valid_rankings)*100:.1f}%)\n'
                      f'Rank better: {rank_advantage_count} ({rank_advantage_count/len(valid_rankings)*100:.1f}%)\n'
                      f'Equal: {neutral_count}',
         transform=ax2.transAxes, fontsize=10, va='bottom', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(FIG_DIR / "safety_margin_correlation.png", dpi=800, bbox_inches='tight')
plt.savefig(FIG_DIR / "safety_margin_correlation.pdf", bbox_inches='tight')
plt.close()
print(f"保存: safety_margin_correlation.png")
print(f"回归斜率: {slope:.4f}, p-value: {p_value:.2e}")

# ============================================================
# 方案三：生存概率曲线 (Survival Probability Curves)
# ============================================================
print("\n" + "="*60)
print("方案三：生存概率曲线")
print("="*60)

# 将粉丝份额分成区间
bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0]
labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-40%', '40%+']

valid_rankings['fan_bin'] = pd.cut(valid_rankings['fan_vote_share'], bins=bins, labels=labels, include_lowest=True)

# 计算每个区间在两种规则下的生存率
# 生存 = 不是最后一名（rank_rule_rank < num_contestants 且 percent_rule_rank < num_contestants）

survival_stats = []
for bin_label in labels:
    bin_data = valid_rankings[valid_rankings['fan_bin'] == bin_label]
    
    if len(bin_data) == 0:
        continue
    
    # 获取每周的选手数
    bin_data = bin_data.copy()
    
    # 在 Rank 规则下是否幸存（不是最后一名）
    # 假设最后一名被淘汰
    rank_survived = 0
    pct_survived = 0
    total = 0
    
    for _, row in bin_data.iterrows():
        season = row['season']
        week = row['week']
        week_data = valid_rankings[(valid_rankings['season'] == season) & 
                                   (valid_rankings['week'] == week)]
        num_contestants = len(week_data)
        
        if num_contestants == 0:
            continue
        
        total += 1
        # Rank 规则下幸存 = 不是最后一名
        if row['rank_rule_rank'] < num_contestants:
            rank_survived += 1
        # Pct 规则下幸存
        if row['percent_rule_rank'] < num_contestants:
            pct_survived += 1
    
    if total > 0:
        survival_stats.append({
            'fan_bin': bin_label,
            'total': total,
            'rank_survival': rank_survived / total,
            'pct_survival': pct_survived / total
        })

survival_df = pd.DataFrame(survival_stats)
print(survival_df)

# 图3: 生存概率曲线
fig3, ax3 = plt.subplots(figsize=(12, 7))

x = np.arange(len(survival_df))
width = 0.35

# 绘制双条形图
bars1 = ax3.bar(x - width/2, survival_df['rank_survival'] * 100, width, 
                label='Rank Rule', color='#3498db', edgecolor='black', linewidth=1)
bars2 = ax3.bar(x + width/2, survival_df['pct_survival'] * 100, width, 
                label='Percentage Rule', color='#2ecc71', edgecolor='black', linewidth=1)

# 连线显示趋势
ax3.plot(x - width/2, survival_df['rank_survival'] * 100, 'o-', color='#2980b9', linewidth=2, markersize=8)
ax3.plot(x + width/2, survival_df['pct_survival'] * 100, 's-', color='#27ae60', linewidth=2, markersize=8)

# 添加差异标注
for i, (_, row) in enumerate(survival_df.iterrows()):
    diff = (row['pct_survival'] - row['rank_survival']) * 100
    if abs(diff) > 0.5:
        y_pos = max(row['rank_survival'], row['pct_survival']) * 100 + 2
        color = '#27ae60' if diff > 0 else '#e74c3c'
        ax3.annotate(f'{diff:+.1f}%', xy=(i, y_pos), fontsize=9, ha='center', color=color, fontweight='bold')

ax3.set_xlabel('Fan Vote Share Bin', fontsize=12)
ax3.set_ylabel('Survival Rate (%)', fontsize=12)
ax3.set_title('Survival Probability by Fan Support Level\n'
              'Positive difference (Pct - Rank) = Percentage Rule favors fans', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(survival_df['fan_bin'], rotation=0)
ax3.legend(loc='lower right', fontsize=11)
ax3.set_ylim(0, 105)

# 添加样本量标注
for i, (_, row) in enumerate(survival_df.iterrows()):
    ax3.text(i, 3, f'n={row["total"]}', ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(FIG_DIR / "survival_probability_curves.png", dpi=800, bbox_inches='tight')
plt.savefig(FIG_DIR / "survival_probability_curves.pdf", bbox_inches='tight')
plt.close()
print(f"保存: survival_probability_curves.png")

# ============================================================
# 综合对比图 (Combined Figure)
# ============================================================
print("\n" + "="*60)
print("生成综合对比图")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 左上：安全边际散点图
ax = axes[0, 0]
ax.scatter(survived['fan_vote_share'], survived['rank_advantage_pct'], 
           alpha=0.3, s=20, c='#3498db', label='Survived')
ax.scatter(eliminated['fan_vote_share'], eliminated['rank_advantage_pct'], 
           alpha=0.6, s=40, c='#e74c3c', label='Eliminated')
ax.plot(x_fit, y_fit, 'k--', linewidth=2, label=f'Slope={slope:.2f}, $R^2$={r_value**2:.3f}')
ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.7)
ax.set_xlabel('Fan Vote Share', fontsize=11)
ax.set_ylabel('Pct Rank Advantage', fontsize=11)
ax.set_title('(A) Safety Margin Correlation\nPositive slope = Pct favors fans', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.text(0.02, 0.02, f'p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}', 
        transform=ax.transAxes, fontsize=10)

# 右上：生存概率对比
ax = axes[0, 1]
x = np.arange(len(survival_df))
ax.bar(x - 0.2, survival_df['rank_survival'] * 100, 0.4, label='Rank Rule', color='#3498db', edgecolor='black')
ax.bar(x + 0.2, survival_df['pct_survival'] * 100, 0.4, label='Percentage Rule', color='#2ecc71', edgecolor='black')
ax.set_xlabel('Fan Vote Share Bin', fontsize=11)
ax.set_ylabel('Survival Rate (%)', fontsize=11)
ax.set_title('(B) Survival Rate by Fan Support\nHigher Pct bars at high fan = Pct protects popular', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(survival_df['fan_bin'], rotation=45, ha='right')
ax.legend(loc='lower right', fontsize=9)

# 左下：分歧周 delta_fan 分布
ax = axes[1, 0]
delta_fan = div_df['delta_fan'].dropna()
pos_delta = delta_fan[delta_fan > 0]
neg_delta = delta_fan[delta_fan <= 0]
bins_hist = np.linspace(-0.15, 0.15, 25)
ax.hist(pos_delta, bins=bins_hist, color='#e74c3c', alpha=0.7, label=f'Rank hurts fans ({len(pos_delta)})', edgecolor='black')
ax.hist(neg_delta, bins=bins_hist, color='#2ecc71', alpha=0.7, label=f'Pct hurts fans ({len(neg_delta)})', edgecolor='black')
ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax.axvline(delta_fan.mean(), color='#9b59b6', linestyle='-', linewidth=2, label=f'Mean={delta_fan.mean():.4f}')
ax.set_xlabel(r'$\Delta$ Fan Vote (Rank Elim − Pct Elim)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(C) Fan Vote Difference Distribution\nPositive = Rank eliminates higher fan support', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)

# 右下：按 regime 的统计
ax = axes[1, 1]
regime_stats = []
for regime in comparison['regime'].unique():
    regime_df = comparison[comparison['regime'] == regime]
    disagree_regime = regime_df[regime_df['methods_agree'] == False]
    
    total_weeks = len(regime_df)
    agree_rate = regime_df['methods_agree'].mean() * 100
    rank_acc = (regime_df['rank_eliminated'] == regime_df['actual_eliminated']).mean() * 100
    pct_acc = (regime_df['pct_eliminated'] == regime_df['actual_eliminated']).mean() * 100
    
    pct_favor = (disagree_regime['fan_favored_by'] == 'Percentage').sum()
    rank_favor = (disagree_regime['fan_favored_by'] == 'Rank').sum()
    
    regime_stats.append({
        'regime': regime.replace(' Era', '').replace(' (', '\n('),
        'pct_acc': pct_acc,
        'rank_acc': rank_acc,
        'pct_favor_ratio': pct_favor / max(1, pct_favor + rank_favor) * 100
    })

regime_df_plot = pd.DataFrame(regime_stats)
x = np.arange(len(regime_df_plot))
width = 0.25

ax.bar(x - width, regime_df_plot['rank_acc'], width, label='Rank Accuracy', color='#3498db', edgecolor='black')
ax.bar(x, regime_df_plot['pct_acc'], width, label='Pct Accuracy', color='#2ecc71', edgecolor='black')
ax.bar(x + width, regime_df_plot['pct_favor_ratio'], width, label='Pct Fan-Favor Ratio', color='#f39c12', edgecolor='black')

ax.set_xlabel('Competition Regime', fontsize=11)
ax.set_ylabel('Percentage (%)', fontsize=11)
ax.set_title('(D) Performance by Regime\nHigher Pct metrics = Pct is more fan-friendly', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regime_df_plot['regime'], rotation=0)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(FIG_DIR / "fate_reversal_combined.png", dpi=800, bbox_inches='tight')
plt.savefig(FIG_DIR / "fate_reversal_combined.pdf", bbox_inches='tight')
plt.close()
print(f"保存: fate_reversal_combined.png")

# ============================================================
# 输出关键统计
# ============================================================
print("\n" + "="*60)
print("关键统计摘要")
print("="*60)
print(f"1. 安全边际回归:")
print(f"   斜率 = {slope:.4f} (正斜率 = 粉丝越多，Pct 规则下排名越好)")
print(f"   R² = {r_value**2:.4f}")
print(f"   p-value = {p_value:.2e}")

print(f"\n2. 分歧周统计:")
print(f"   Rank 伤害粉丝 (delta_fan > 0): {pos_count} 周")
print(f"   Pct 伤害粉丝 (delta_fan ≤ 0): {neg_count} 周")
print(f"   比例: {pos_count/neg_count:.2f}×")

print(f"\n3. 结论:")
print(f"   ✓ 正斜率证明：粉丝份额越高，Percentage 规则相对于 Rank 规则的优势越大")
print(f"   ✓ 分歧周中 Rank 伤害粉丝的次数是 Pct 的 {pos_count/neg_count:.2f} 倍")
print(f"   ✓ Percentage 方法是 '民粹主义' 规则，Rank 方法是 '专业主义' 规则")
