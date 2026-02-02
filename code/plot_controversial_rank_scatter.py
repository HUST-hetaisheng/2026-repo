"""
Model 2 可视化：争议选手分析
核心问题：裁判评分最差的选手，为何能获得好名次？
X轴：裁判"最后一�?的周数占比（越高=裁判评价越差�?
Y轴：最终名次（归一化，0=冠军�?=首轮淘汰�?
争议选手 = 左上角（裁判差，但名次好�?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

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

# 获取每个选手的最终名次和赛季人数
placement_info = df_raw[['celebrity_name', 'season', 'placement']].copy()
season_counts = df_raw.groupby('season')['celebrity_name'].count().reset_index()
season_counts.columns = ['season', 'n_contestants']
placement_info = placement_info.merge(season_counts, on='season')
placement_info['placement_norm'] = (placement_info['placement'] - 1) / (placement_info['n_contestants'] - 1)

def compute_controversy_stats(df_fan):
    """计算每个选手的争议指标（排除已淘汰选手�?""
    results = []
    
    for (season, celebrity), group in df_fan.groupby(['season', 'celebrity_name']):
        # 只统计该选手有有效judge分数的周（未被淘汰）
        valid_group = group[group['judge_total'] > 0]
        weeks_played = len(valid_group)
        
        if weeks_played == 0:
            continue
            
        judge_last_count = 0
        judge_bottom2_count = 0
        fan_top2_count = 0
        
        rank_diffs = []
        share_diffs = []
        
        for _, row in valid_group.iterrows():
            week = row['week']
            # 只选取该周有有效分数的选手（排除已淘汰�?
            week_data = df_fan[(df_fan['season'] == season) & 
                               (df_fan['week'] == week) & 
                               (df_fan['judge_total'] > 0)]
            n = len(week_data)
            
            if n < 2:
                continue
            
            judge_scores = week_data['judge_total'].values
            fan_shares = week_data['fan_vote_share'].values
            names = week_data['celebrity_name'].values
            
            judge_rank = rankdata(-judge_scores, method='min')
            fan_rank = rankdata(-fan_shares, method='min')
            
            idx = list(names).index(celebrity)
            jr = judge_rank[idx]
            fr = fan_rank[idx]
            
            # 统计
            if jr == n:
                judge_last_count += 1
            if jr >= n - 1:
                judge_bottom2_count += 1
            if fr <= 2:
                fan_top2_count += 1
            
            # Rank差异（归一化）
            jr_norm = (jr - 1) / (n - 1)
            fr_norm = (fr - 1) / (n - 1)
            rank_diffs.append(jr_norm - fr_norm)
            
            # Share差异
            judge_share = row['judge_total'] / week_data['judge_total'].sum()
            fan_share = row['fan_vote_share']
            share_diffs.append(fan_share - judge_share)
        
        results.append({
            'season': season,
            'celebrity_name': celebrity,
            'weeks_played': weeks_played,
            'judge_last_count': judge_last_count,
            'judge_bottom2_count': judge_bottom2_count,
            'fan_top2_count': fan_top2_count,
            'judge_last_pct': judge_last_count / weeks_played if weeks_played > 0 else 0,
            'judge_bottom2_pct': judge_bottom2_count / weeks_played if weeks_played > 0 else 0,
            'avg_rank_diff': np.mean(rank_diffs) if rank_diffs else 0,
            'avg_share_diff': np.mean(share_diffs) if share_diffs else 0,
        })
    
    return pd.DataFrame(results)

# 计算争议统计
stats_df = compute_controversy_stats(df_fan)

# 合并最终名�?
stats_df = stats_df.merge(
    placement_info[['celebrity_name', 'season', 'placement', 'placement_norm', 'n_contestants']],
    on=['season', 'celebrity_name'],
    how='left'
)

print("=" * 70)
print("争议选手详细统计")
print("=" * 70)

for name, info in CONTROVERSIAL.items():
    subset = stats_df[(stats_df['celebrity_name'] == name) & (stats_df['season'] == info['season'])]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f"\n{name} (Season {info['season']}):")
        print(f"  参赛周数: {int(row['weeks_played'])}")
        print(f"  裁判最后一�? {int(row['judge_last_count'])}�?({row['judge_last_pct']:.0%})")
        print(f"  裁判倒数两名: {int(row['judge_bottom2_count'])}�?({row['judge_bottom2_pct']:.0%})")
        print(f"  粉丝前两�? {int(row['fan_top2_count'])}�?)
        print(f"  平均Rank�?J-F): {row['avg_rank_diff']:+.2f}")
        print(f"  最终名�? {int(row['placement'])}/{int(row['n_contestants'])}")

# ============================================================
# �?: 裁判倒数比例 vs 最终名�?
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# -------------------- 左图：裁判倒数比例 vs 名次 --------------------
ax1 = axes[0]

# 普通选手
normal_mask = ~stats_df['celebrity_name'].isin(CONTROVERSIAL.keys())
ax1.scatter(stats_df.loc[normal_mask, 'judge_bottom2_pct'],
            stats_df.loc[normal_mask, 'placement_norm'],
            c='gray', alpha=0.4, s=20, edgecolors='none')

# 争议选手
for name, info in CONTROVERSIAL.items():
    mask = (stats_df['celebrity_name'] == name) & (stats_df['season'] == info['season'])
    subset = stats_df[mask]
    if len(subset) > 0:
        ax1.scatter(subset['judge_bottom2_pct'], subset['placement_norm'],
                   c=info['color'], marker=info['marker'], s=100,
                   edgecolors='black', linewidths=1.2, zorder=10,
                   label=f"{name} (S{info['season']})")

# 高亮争议区域（右上：裁判差，名次好）
ax1.fill_between([0.4, 1.05], [-0.05, -0.05], [0.4, 0.4], alpha=0.1, color='red')
ax1.text(0.7, 0.2, 'Controversial\nZone', ha='center', fontsize=10, 
         color='darkred', fontstyle='italic')

ax1.set_xlabel('Judge Bottom-2 Rate\n(% of weeks ranked in bottom 2 by judges)', fontsize=11)
ax1.set_ylabel('Final Placement (normalized)\n�?Winner | First Out �?, fontsize=11)
ax1.set_title('(a) Judge Performance vs Final Placement', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# -------------------- 右图：Rank差异 vs 名次（对称X轴） --------------------
ax2 = axes[1]

# 计算对称X轴范�?
max_diff = max(abs(stats_df['avg_rank_diff'].min()), abs(stats_df['avg_rank_diff'].max()))
xlim = (-max_diff * 1.15, max_diff * 1.15)

# 普通选手
ax2.scatter(stats_df.loc[normal_mask, 'avg_rank_diff'],
            stats_df.loc[normal_mask, 'placement_norm'],
            c='gray', alpha=0.4, s=20, edgecolors='none')

# 争议选手
for name, info in CONTROVERSIAL.items():
    mask = (stats_df['celebrity_name'] == name) & (stats_df['season'] == info['season'])
    subset = stats_df[mask]
    if len(subset) > 0:
        ax2.scatter(subset['avg_rank_diff'], subset['placement_norm'],
                   c=info['color'], marker=info['marker'], s=100,
                   edgecolors='black', linewidths=1.2, zorder=10,
                   label=f"{name} (S{info['season']})")

# 零线
ax2.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

# 高亮争议区域
ax2.fill_between([0, xlim[1]], [-0.05, -0.05], [0.4, 0.4], alpha=0.1, color='red')
ax2.text(xlim[1]*0.5, 0.2, 'Controversial\nZone', ha='center', fontsize=10,
         color='darkred', fontstyle='italic')

ax2.set_xlabel('Avg (Judge Rank �?Fan Rank), normalized\n�?Fan favored | Judge favored �?, fontsize=11)
ax2.set_ylabel('Final Placement (normalized)\n�?Winner | First Out �?, fontsize=11)
ax2.set_title('(b) Rank Difference vs Final Placement', fontsize=12, fontweight='bold')
ax2.set_xlim(xlim)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('d:/2026-repo/figures/controversial_analysis.png', dpi=1000, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("图已保存: figures/controversial_analysis.png")
print("=" * 70)

# ============================================================
# 自动检测争议选手
# ============================================================
print("\n" + "=" * 70)
print("自动检测争议选手（裁判倒数2�?> 40% �?名次在前50%�?)
print("=" * 70)

auto_controversial = stats_df[
    (stats_df['judge_bottom2_pct'] > 0.4) & 
    (stats_df['placement_norm'] < 0.5)
].sort_values('judge_bottom2_pct', ascending=False)

for _, row in auto_controversial.head(15).iterrows():
    print(f"{row['celebrity_name']} (S{int(row['season'])}): "
          f"倒数2�?{row['judge_bottom2_pct']:.0%}, "
          f"名次={int(row['placement'])}/{int(row['n_contestants'])}")

# 保存
stats_df.to_csv('d:/2026-repo/data/controversial_stats.csv', index=False)
print(f"\n完整数据已保�? data/controversial_stats.csv")
