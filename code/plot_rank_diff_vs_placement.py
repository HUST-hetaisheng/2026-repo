"""
Model 2 可视化：争议选手检�?
X轴：裁判Rank - 粉丝Rank（正�?裁判评价差于粉丝=争议�?
Y轴：最终名次（归一化，0=冠军�?=首轮淘汰�?

争议选手特征：X值大（裁判差、粉丝好），但Y值小（最终名次好�?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# 读取数据
fan_df = pd.read_csv('d:/2026-repo/data/fan_vote_results_final.csv')
orig_df = pd.read_csv('d:/2026-repo/data/2026_MCM_Problem_C_Data_Cleaned.csv')

# 争议选手列表
CONTROVERSIAL = {
    'Jerry Rice': {'season': 2, 'color': 'red', 'marker': 's'},
    'Billy Ray Cyrus': {'season': 4, 'color': 'orange', 'marker': '^'},
    'Bristol Palin': {'season': 11, 'color': 'purple', 'marker': 'D'},
    'Bobby Bones': {'season': 27, 'color': 'green', 'marker': 'p'},
}

# ============================================================
# 计算每周的Rank差异
# ============================================================
def compute_weekly_rank_diff(fan_df):
    """计算每周：裁判Rank - 粉丝Rank"""
    results = []
    
    for (season, week), group in fan_df.groupby(['season', 'week']):
        n = len(group)
        if n < 2:
            continue
        
        # Rank�?=最好，n=最�?
        judge_rank = rankdata(-group['judge_total'].values, method='average')
        fan_rank = rankdata(-group['fan_vote_share'].values, method='average')
        
        # Share（归一化）
        judge_share = group['judge_total'].values / group['judge_total'].sum()
        fan_share = group['fan_vote_share'].values
        
        for i, (_, row) in enumerate(group.iterrows()):
            results.append({
                'season': season,
                'week': week,
                'celebrity_name': row['celebrity_name'],
                'judge_rank': judge_rank[i],
                'fan_rank': fan_rank[i],
                'rank_diff': judge_rank[i] - fan_rank[i],  # �?裁判差于粉丝
                'judge_share': judge_share[i],
                'fan_share': fan_share[i],
                'share_diff': fan_share[i] - judge_share[i],  # �?粉丝高于裁判
                'n_contestants': n
            })
    
    return pd.DataFrame(results)

weekly_df = compute_weekly_rank_diff(fan_df)

# 计算每个选手的赛季平�?
contestant_stats = weekly_df.groupby(['season', 'celebrity_name']).agg({
    'rank_diff': 'mean',      # 平均Rank差异
    'share_diff': 'mean',     # 平均Share差异
    'week': 'count'           # 参赛周数
}).reset_index()
contestant_stats.columns = ['season', 'celebrity_name', 'avg_rank_diff', 'avg_share_diff', 'weeks']

# ============================================================
# 获取最终名次并归一�?
# ============================================================
# 计算每赛季总人�?
season_size = orig_df.groupby('season')['celebrity_name'].count().reset_index()
season_size.columns = ['season', 'n_total']

# 合并最终名�?
placement_df = orig_df[['celebrity_name', 'season', 'placement']].copy()
contestant_stats = contestant_stats.merge(placement_df, on=['season', 'celebrity_name'], how='left')
contestant_stats = contestant_stats.merge(season_size, on='season', how='left')

# 归一化名次：0=冠军�?=最后一�?
contestant_stats['placement_norm'] = (contestant_stats['placement'] - 1) / (contestant_stats['n_total'] - 1)

print("=" * 70)
print("争议选手统计")
print("=" * 70)

for name, info in CONTROVERSIAL.items():
    subset = contestant_stats[(contestant_stats['celebrity_name'] == name) & 
                              (contestant_stats['season'] == info['season'])]
    if len(subset) == 0:
        print(f"\n{name}: 未找到数�?)
        continue
    
    row = subset.iloc[0]
    print(f"\n{name} (Season {info['season']}):")
    print(f"  最终名�? {int(row['placement'])}/{int(row['n_total'])} (归一�? {row['placement_norm']:.2f})")
    print(f"  平均Rank�?裁判-粉丝): {row['avg_rank_diff']:+.2f}")
    print(f"  平均Share�?粉丝-裁判): {row['avg_share_diff']:+.2%}")
    print(f"  参赛周数: {int(row['weeks'])}")

# ============================================================
# �?: Rank差异 vs 最终名�?
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ----- 左图：Rank差异 -----
ax1 = axes[0]

# 普通选手
normal_mask = ~contestant_stats['celebrity_name'].isin(CONTROVERSIAL.keys())
ax1.scatter(contestant_stats.loc[normal_mask, 'avg_rank_diff'],
            contestant_stats.loc[normal_mask, 'placement_norm'],
            alpha=0.5, c='gray', s=40, label='Other contestants')

# 争议选手
for name, info in CONTROVERSIAL.items():
    mask = (contestant_stats['celebrity_name'] == name) & (contestant_stats['season'] == info['season'])
    subset = contestant_stats[mask]
    if len(subset) > 0:
        x = subset['avg_rank_diff'].values[0]
        y = subset['placement_norm'].values[0]
        ax1.scatter(x, y, c=info['color'], marker=info['marker'], s=250,
                   edgecolors='black', linewidths=2, zorder=10)
        ax1.annotate(f"{name}\n(S{info['season']})", (x, y),
                    xytext=(8, -5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=info['color'])

# 分界�?
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.3)

# 标注区域
ax1.fill_betweenx([0, 0.3], [1, 1], [ax1.get_xlim()[1] if ax1.get_xlim()[1] > 1 else 4, 4], 
                  alpha=0.15, color='red')
ax1.text(2.5, 0.15, '�?CONTROVERSIAL\nBad Judge Rank\nGood Final Place', 
         ha='center', fontsize=10, color='darkred', fontweight='bold')

ax1.set_xlabel('Average (Judge Rank - Fan Rank)\n�?Fan Favored | Judge Favored �?, fontsize=11)
ax1.set_ylabel('Final Placement (Normalized)\n0 = Winner, 1 = First Eliminated', fontsize=11)
ax1.set_title('RANK Method: Rank Difference vs Final Placement', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.invert_yaxis()  # 名次好的在上�?

# ----- 右图：Share差异 -----
ax2 = axes[1]

# 普通选手
ax2.scatter(contestant_stats.loc[normal_mask, 'avg_share_diff'],
            contestant_stats.loc[normal_mask, 'placement_norm'],
            alpha=0.5, c='gray', s=40, label='Other contestants')

# 争议选手
for name, info in CONTROVERSIAL.items():
    mask = (contestant_stats['celebrity_name'] == name) & (contestant_stats['season'] == info['season'])
    subset = contestant_stats[mask]
    if len(subset) > 0:
        x = subset['avg_share_diff'].values[0]
        y = subset['placement_norm'].values[0]
        ax2.scatter(x, y, c=info['color'], marker=info['marker'], s=250,
                   edgecolors='black', linewidths=2, zorder=10)
        ax2.annotate(f"{name}\n(S{info['season']})", (x, y),
                    xytext=(8, -5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=info['color'])

# 分界�?
ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.3)

# 标注区域
xlim = ax2.get_xlim()
ax2.fill_betweenx([0, 0.3], [0.05, 0.05], [xlim[1] if xlim[1] > 0.1 else 0.2, 0.2],
                  alpha=0.15, color='red')
ax2.text(0.1, 0.15, '�?CONTROVERSIAL\nHigh Fan Share\nGood Final Place',
         ha='center', fontsize=10, color='darkred', fontweight='bold')

ax2.set_xlabel('Average (Fan Share - Judge Share)\n�?Judge Favored | Fan Favored �?, fontsize=11)
ax2.set_ylabel('Final Placement (Normalized)\n0 = Winner, 1 = First Eliminated', fontsize=11)
ax2.set_title('PERCENTAGE Method: Share Difference vs Final Placement', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('d:/2026-repo/figures/controversial_rank_vs_placement.png', dpi=800, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("图已保存: figures/controversial_rank_vs_placement.png")
print("=" * 70)

# ============================================================
# 自动检测争议选手
# ============================================================
print("\n" + "=" * 70)
print("自动检测的争议选手（Rank�?1 �?名次�?0%�?)
print("=" * 70)

# 条件：裁判Rank明显差于粉丝（差>1），但最终名次好（前50%�?
detected = contestant_stats[
    (contestant_stats['avg_rank_diff'] > 1) & 
    (contestant_stats['placement_norm'] < 0.5)
].sort_values('avg_rank_diff', ascending=False)

for _, row in detected.iterrows():
    print(f"{row['celebrity_name']} (S{int(row['season'])}): "
          f"Rank�?{row['avg_rank_diff']:+.2f}, "
          f"名次={int(row['placement'])}/{int(row['n_total'])}")

print(f"\n共检测到 {len(detected)} 位争议选手")

# 保存
detected.to_csv('d:/2026-repo/data/controversial_detected.csv', index=False)
