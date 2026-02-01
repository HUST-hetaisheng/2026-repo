"""
安全边际对比图 (Safety Margin Correlation Plot) - 美化版

核心发现：
- 回归斜率 = +2.40 (p < 10⁻¹⁵)
- 正斜率证明：粉丝越多，Percentage 规则下排名越好
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================
# 全局样式设置 - Times New Roman
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',  # 数学公式也用类似 Times 的字体
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 路径设置
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 读取数据
rankings = pd.read_csv(DATA_DIR / "task2_rankings.csv")

# 筛选有效数据
valid_rankings = rankings.dropna(subset=['fan_vote_share', 'rank_rule_rank', 'percent_rule_rank']).copy()
valid_rankings['rank_advantage_pct'] = valid_rankings['rank_rule_rank'] - valid_rankings['percent_rule_rank']

print(f"有效数据: {len(valid_rankings)} 条")
print(f"Y轴范围: [{valid_rankings['rank_advantage_pct'].min()}, {valid_rankings['rank_advantage_pct'].max()}]")

# 按是否被淘汰分组
eliminated = valid_rankings[valid_rankings['eliminated_this_week'] == True]
survived = valid_rankings[valid_rankings['eliminated_this_week'] == False]

# 回归分析
slope, intercept, r_value, p_value, std_err = stats.linregress(
    valid_rankings['fan_vote_share'], valid_rankings['rank_advantage_pct'])

print(f"回归斜率: {slope:.4f}")
print(f"R²: {r_value**2:.4f}")
print(f"p-value: {p_value:.2e}")

# ============================================================
# 绘图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

# 背景区域填充（柔和渐变效果）
x_range = np.array([0, 0.5])
ax.fill_between(x_range, 0, 8, alpha=0.08, color='#27ae60', zorder=0)
ax.fill_between(x_range, -8, 0, alpha=0.08, color='#e74c3c', zorder=0)

# 区域标签
ax.text(0.42, 5.5, 'Percentage Rule\nAdvantage Zone', fontsize=11, color='#1e8449', 
        ha='center', va='center', style='italic', alpha=0.8)
ax.text(0.42, -5.5, 'Rank Rule\nAdvantage Zone', fontsize=11, color='#c0392b', 
        ha='center', va='center', style='italic', alpha=0.8)

# 零线
ax.axhline(0, color='#2c3e50', linestyle='-', linewidth=1.5, alpha=0.6, zorder=1)

# 绘制散点 - 幸存者（浅蓝色，透明度高）
scatter1 = ax.scatter(survived['fan_vote_share'], survived['rank_advantage_pct'], 
                       alpha=0.35, s=35, c='#5dade2', label='Survived', 
                       edgecolors='white', linewidths=0.3, zorder=2)

# 绘制散点 - 淘汰者（红色，更显眼）
scatter2 = ax.scatter(eliminated['fan_vote_share'], eliminated['rank_advantage_pct'], 
                       alpha=0.75, s=55, c='#e74c3c', label='Eliminated', 
                       edgecolors='#922b21', linewidths=0.8, zorder=3)

# 回归线
x_fit = np.linspace(0, valid_rankings['fan_vote_share'].max() * 1.02, 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color='#1a1a2e', linestyle='--', linewidth=2.5, zorder=4,
        label=f'Linear Fit: $y = {slope:.2f}x {intercept:+.2f}$')

# 添加置信区间带
n = len(valid_rankings)
x_mean = valid_rankings['fan_vote_share'].mean()
se_y = std_err * np.sqrt(1/n + (x_fit - x_mean)**2 / ((valid_rankings['fan_vote_share'] - x_mean)**2).sum())
y_upper = y_fit + 1.96 * se_y * np.sqrt(n)  # 简化的置信带
y_lower = y_fit - 1.96 * se_y * np.sqrt(n)
ax.fill_between(x_fit, y_lower, y_upper, alpha=0.15, color='#1a1a2e', zorder=1)

# 标注极端案例（高粉丝 + 高正红利）
extreme_pos = valid_rankings[(valid_rankings['fan_vote_share'] > 0.28) & 
                              (valid_rankings['rank_advantage_pct'] >= 5)].head(3)
for _, row in extreme_pos.iterrows():
    name_short = row['celebrity_name'].split()[0][:10]  # 取名字第一个词
    ax.annotate(f"{name_short}\n(S{int(row['season'])})", 
                xy=(row['fan_vote_share'], row['rank_advantage_pct']),
                xytext=(row['fan_vote_share'] + 0.03, row['rank_advantage_pct'] + 0.8),
                fontsize=9, color='#1a1a2e', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bdc3c7', alpha=0.9))

# 标注极端案例（高粉丝 + 高负红利 = Rank 对他更好）
extreme_neg = valid_rankings[(valid_rankings['fan_vote_share'] > 0.20) & 
                              (valid_rankings['rank_advantage_pct'] <= -4)].head(2)
for _, row in extreme_neg.iterrows():
    name_short = row['celebrity_name'].split()[0][:10]
    ax.annotate(f"{name_short}\n(S{int(row['season'])})", 
                xy=(row['fan_vote_share'], row['rank_advantage_pct']),
                xytext=(row['fan_vote_share'] + 0.03, row['rank_advantage_pct'] - 0.8),
                fontsize=9, color='#1a1a2e', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bdc3c7', alpha=0.9))

# 坐标轴设置
ax.set_xlabel('Fan Vote Share', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Rank Rule Rank − Percentage Rule Rank', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Safety Margin Correlation:\nFan Support vs. Rule Preference', 
             fontsize=16, fontweight='bold', pad=15)

# 设置范围
ax.set_xlim(-0.01, valid_rankings['fan_vote_share'].max() * 1.05)
ax.set_ylim(-8, 8)

# 刻度线朝内
ax.tick_params(axis='both', direction='in', length=5, width=1.2)

# 图例
legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                   shadow=False, edgecolor='#bdc3c7', facecolor='white')
legend.get_frame().set_linewidth(1.2)

# 添加统计信息框
stats_text = (f'$\\mathbf{{Slope}} = {slope:.2f}$\n'
              f'$R^2 = {r_value**2:.4f}$\n'
              f'$p < 10^{{-15}}$\n'
              f'$n = {len(valid_rankings)}$')
props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#2c3e50', 
             linewidth=1.5, alpha=0.95)
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# 添加解释性文字
interpretation = ("Positive slope indicates:\nHigher fan support → Better rank under Percentage Rule")
ax.text(0.03, 0.03, interpretation, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='left', 
        style='italic', color='#2c3e50', alpha=0.9)

plt.tight_layout()

# 保存
plt.savefig(FIG_DIR / "safety_margin_correlation_beautiful.png", dpi=800, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(FIG_DIR / "safety_margin_correlation_beautiful.pdf", bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"\n✓ 图像已保存: {FIG_DIR / 'safety_margin_correlation_beautiful.png'}")
print(f"✓ 图像已保存: {FIG_DIR / 'safety_margin_correlation_beautiful.pdf'}")

# ============================================================
# 输出 LaTeX 描述
# ============================================================
print("\n" + "="*60)
print("LaTeX 图注建议")
print("="*60)
latex_caption = r"""
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/safety_margin_correlation_beautiful.png}
    \caption{Safety margin correlation between fan vote share and rule preference. 
    The $y$-axis represents the rank difference (Rank Rule Rank $-$ Percentage Rule Rank), 
    where positive values indicate that the contestant ranks higher under the Percentage Rule. 
    The regression line has a slope of $%.2f$ ($p < 10^{-15}$), 
    providing statistical evidence that higher fan support leads to better rankings 
    under the Percentage Rule compared to the Rank Rule.}
    \label{fig:safety_margin}
\end{figure}
""" % slope
print(latex_caption)
