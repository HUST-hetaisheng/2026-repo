"""
Model 2 可视化：两种方法的差�?vs 最终名�?
左图：裁判Rank - 粉丝Rank vs 最终名�?
右图：裁判Percent - 粉丝Percent vs 最终名�?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

# 读取数据
df_fan = pd.read_csv('d:/2026-repo/data/fan_vote_results_final.csv')
df_raw = pd.read_csv('d:/2026-repo/data/2026_MCM_Problem_C_Data_Cleaned.csv')

# 争议选手列表
CONTROVERSIAL = {
    'Jerry Rice': {'season': 2, 'color': '#E41A1C', 'marker': 's'},
    'Billy Ray Cyrus': {'season': 4, 'color': '#FF7F00', 'marker': '^'},
    'Bristol Palin': {'season': 11, 'color': '#984EA3', 'marker': 'D'},
    'Bobby Bones': {'season': 27, 'color': '#4DAF4A', 'marker': 'p'},
}

# 获取最终名次信�?
placement_info = df_raw[['celebrity_name', 'season', 'placement']].copy()
season_counts = df_raw.groupby('season')['celebrity_name'].count().reset_index()
season_counts.columns = ['season', 'n_contestants']
placement_info = placement_info.merge(season_counts, on='season')
# 归一化：0=冠军�?=首轮淘汰
placement_info['placement_norm'] = (placement_info['placement'] - 1) / (placement_info['n_contestants'] - 1)

def compute_weekly_metrics(df_fan):
    """计算每周每个选手的Rank差和Percent差（排除已淘汰选手�?""
    results = []
    
    for (season, week), group in df_fan.groupby(['season', 'week']):
        # 排除已淘汰选手（judge_total <= 0�?
        valid = group[group['judge_total'] > 0].copy()
        n = len(valid)
        
        if n < 2:
            continue
        
        # 计算Rank�?=最好，n=最差）
        judge_rank = rankdata(-valid['judge_total'].values, method='average')
        fan_rank = rankdata(-valid['fan_vote_share'].values, method='average')
        
        # 计算Percent（份额）
        judge_pct = valid['judge_total'].values / valid['judge_total'].sum()
        fan_pct = valid['fan_vote_share'].values  # 已经是份�?
        
        for i, (_, row) in enumerate(valid.iterrows()):
            results.append({
                'season': season,
                'week': week,
                'celebrity_name': row['celebrity_name'],
                'judge_rank': judge_rank[i],
                'fan_rank': fan_rank[i],
                'judge_pct': judge_pct[i],
                'fan_pct': fan_pct[i],
                'n_contestants': n,
                # Rank差：�?裁判比粉丝评价差
                'rank_diff': judge_rank[i] - fan_rank[i],
                # Percent差：�?裁判比粉丝评价差（裁判份额低�?
                'pct_diff': fan_pct[i] - judge_pct[i],
            })
    
    return pd.DataFrame(results)

# 计算每周指标
weekly_df = compute_weekly_metrics(df_fan)

# 按选手-赛季汇总（平均值）
contestant_stats = weekly_df.groupby(['season', 'celebrity_name']).agg({
    'rank_diff': 'mean',
    'pct_diff': 'mean',
    'judge_rank': 'mean',
    'fan_rank': 'mean',
    'judge_pct': 'mean',
    'fan_pct': 'mean',
    'n_contestants': 'mean',
    'week': 'count'
}).reset_index()
contestant_stats.columns = ['season', 'celebrity_name', 'avg_rank_diff', 'avg_pct_diff',
                            'avg_judge_rank', 'avg_fan_rank', 'avg_judge_pct', 'avg_fan_pct',
                            'avg_n', 'weeks']

# 归一化Rank差（除以平均选手�?1�?
contestant_stats['rank_diff_norm'] = contestant_stats['avg_rank_diff'] / (contestant_stats['avg_n'] - 1)

# 合并最终名�?
contestant_stats = contestant_stats.merge(
    placement_info[['celebrity_name', 'season', 'placement', 'placement_norm', 'n_contestants']],
    on=['season', 'celebrity_name'],
    how='left'
)

# ============================================================
# 打印争议选手统计
# ============================================================
print("=" * 70)
print("争议选手统计（修复后�?)
print("=" * 70)

for name, info in CONTROVERSIAL.items():
    subset = contestant_stats[(contestant_stats['celebrity_name'] == name) & 
                              (contestant_stats['season'] == info['season'])]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f"\n{name} (Season {info['season']}):")
        print(f"  参赛周数: {int(row['weeks'])}")
        print(f"  平均裁判Rank: {row['avg_judge_rank']:.2f}, 平均粉丝Rank: {row['avg_fan_rank']:.2f}")
        print(f"  Rank�?(J-F): {row['avg_rank_diff']:+.2f} (归一�? {row['rank_diff_norm']:+.3f})")
        print(f"  平均裁判%: {row['avg_judge_pct']:.1%}, 平均粉丝%: {row['avg_fan_pct']:.1%}")
        print(f"  Pct�?(F-J): {row['avg_pct_diff']:+.1%}")
        print(f"  最终名�? {int(row['placement'])}/{int(row['n_contestants'])} (归一�? {row['placement_norm']:.2f})")

# ============================================================
# 绘图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 计算对称X轴范�?
max_rank = max(abs(contestant_stats['rank_diff_norm'].min()), 
               abs(contestant_stats['rank_diff_norm'].max()))
xlim_rank = (-max_rank * 1.15, max_rank * 1.15)

max_pct = max(abs(contestant_stats['avg_pct_diff'].min()), 
              abs(contestant_stats['avg_pct_diff'].max()))
xlim_pct = (-max_pct * 1.15, max_pct * 1.15)

# -------------------- 左图：Rank�?vs 名次 --------------------
ax1 = axes[0]

# 普通选手
normal_mask = ~contestant_stats['celebrity_name'].isin(CONTROVERSIAL.keys())
ax1.scatter(contestant_stats.loc[normal_mask, 'rank_diff_norm'],
            contestant_stats.loc[normal_mask, 'placement_norm'],
            c='gray', alpha=0.4, s=20, edgecolors='none')

# 争议选手
for name, info in CONTROVERSIAL.items():
    mask = (contestant_stats['celebrity_name'] == name) & (contestant_stats['season'] == info['season'])
    subset = contestant_stats[mask]
    if len(subset) > 0:
        ax1.scatter(subset['rank_diff_norm'], subset['placement_norm'],
                   c=info['color'], marker=info['marker'], s=80,
                   edgecolors='black', linewidths=1.2, zorder=10,
                   label=f"{name} (S{info['season']})")

# 零线和争议区�?
ax1.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.fill_between([0, xlim_rank[1]], [-0.05, -0.05], [0.35, 0.35], 
                  alpha=0.1, color='red', zorder=0)
ax1.text(xlim_rank[1]*0.55, 0.15, 'Controversial\nZone', ha='center', fontsize=10,
         color='darkred', fontstyle='italic')

ax1.set_xlabel('(Judge Rank �?Fan Rank) / (N�?)\n�?Fan Favored | Judge Favored �?, fontsize=11)
ax1.set_ylabel('Final Placement (normalized)\n0 = Winner, 1 = First Eliminated', fontsize=11)
ax1.set_title('(a) RANK Method Difference vs Final Placement', fontsize=12, fontweight='bold')
ax1.set_xlim(xlim_rank)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# -------------------- 右图：Pct�?vs 名次 --------------------
ax2 = axes[1]

# 普通选手
ax2.scatter(contestant_stats.loc[normal_mask, 'avg_pct_diff'],
            contestant_stats.loc[normal_mask, 'placement_norm'],
            c='gray', alpha=0.4, s=20, edgecolors='none')

# 争议选手
for name, info in CONTROVERSIAL.items():
    mask = (contestant_stats['celebrity_name'] == name) & (contestant_stats['season'] == info['season'])
    subset = contestant_stats[mask]
    if len(subset) > 0:
        ax2.scatter(subset['avg_pct_diff'], subset['placement_norm'],
                   c=info['color'], marker=info['marker'], s=80,
                   edgecolors='black', linewidths=1.2, zorder=10,
                   label=f"{name} (S{info['season']})")

# 零线和争议区�?
ax2.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.fill_between([0, xlim_pct[1]], [-0.05, -0.05], [0.35, 0.35], 
                  alpha=0.1, color='red', zorder=0)
ax2.text(xlim_pct[1]*0.55, 0.15, 'Controversial\nZone', ha='center', fontsize=10,
         color='darkred', fontstyle='italic')

ax2.set_xlabel('(Fan % �?Judge %)\n�?Judge Favored | Fan Favored �?, fontsize=11)
ax2.set_ylabel('Final Placement (normalized)\n0 = Winner, 1 = First Eliminated', fontsize=11)
ax2.set_title('(b) PERCENTAGE Method Difference vs Final Placement', fontsize=12, fontweight='bold')
ax2.set_xlim(xlim_pct)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('d:/2026-repo/figures/rank_pct_vs_placement.png', dpi=800, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("图已保存: figures/rank_pct_vs_placement.png")
print("=" * 70)

# ============================================================
# 自动检测争议选手
# ============================================================
print("\n" + "=" * 70)
print("争议选手检测（Rank�?> 0.1 �?名次�?0%�?)
print("=" * 70)

auto_controversial = contestant_stats[
    (contestant_stats['rank_diff_norm'] > 0.1) & 
    (contestant_stats['placement_norm'] < 0.4)
].sort_values('rank_diff_norm', ascending=False)

for _, row in auto_controversial.head(15).iterrows():
    print(f"{row['celebrity_name']} (S{int(row['season'])}): "
          f"Rank�?{row['rank_diff_norm']:+.3f}, Pct�?{row['avg_pct_diff']:+.1%}, "
          f"名次={int(row['placement'])}/{int(row['n_contestants'])}")

# 保存
contestant_stats.to_csv('d:/2026-repo/data/rank_pct_analysis.csv', index=False)
print(f"\n数据已保�? data/rank_pct_analysis.csv")
