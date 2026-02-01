"""
Counterfactual Survival Trajectory Plot
Compare Rank vs Percentage method Safety Margin for 4 controversial contestants

Safety Margin (normalized 0-1):
- 0 = Elimination line (Bottom-2 / lowest score)
- 1 = Safest position (Best performer that week)

X-axis: Week (symmetric range)
Y-axis: Normalized Safety Margin (0-1, same scale for both methods)

Data Sources:
- fan_vote_results_final.csv: 包含 fan_vote_share, judge_total, week, season
- 2026_MCM_Problem_C_Data.csv: 包含 placement 等基础信息
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


# === Controversial contestants ===
CONTROVERSIAL = {
    "Jerry Rice": {"season": 2},
    "Billy Ray Cyrus": {"season": 4},
    "Bristol Palin": {"season": 11},
    "Bobby Bones": {"season": 27},
}


def load_data():
    """加载fan_vote_results_final.csv (已有每周的judge_total和fan_vote_share)"""
    df = pd.read_csv("d:/2026-repo/data/fan_vote_results_final.csv")
    return df


def compute_weekly_metrics(df, season):
    """
    对于每周,计算每位选手的归一化Safety Margin (0 = 淘汰�? 1 = 最安全)
    
    Rank Method: 
      - judge_rank + fan_rank (越小越安�?
      - 归一�? (max_combined - my_combined) / (max_combined - min_combined)
    
    Percentage Method:
      - judge_share + fan_share (越大越安�?  
      - 归一�? (my_combined - min_combined) / (max_combined - min_combined)
    """
    season_df = df[df['season'] == season].copy()
    
    # 只保留有效选手(有裁判分)
    season_df = season_df[season_df['judge_total'] > 0].copy()
    
    weeks = sorted(season_df['week'].unique())
    
    results = []
    
    for week in weeks:
        week_df = season_df[season_df['week'] == week].copy()
        
        if len(week_df) < 2:
            continue
        
        n = len(week_df)
        
        # === Rank Method ===
        # 裁判排名: 分数越高排名越好(1=最�?
        week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False, method='min')
        # 粉丝排名: 份额越高排名越好(1=最�?
        week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False, method='min')
        # 综合排名: rank越低越安�?
        week_df['combined_rank'] = week_df['judge_rank'] + week_df['fan_rank']
        
        # 归一�? 0 = 最危险(最高combined_rank), 1 = 最安全(最低combined_rank)
        max_rank = week_df['combined_rank'].max()
        min_rank = week_df['combined_rank'].min()
        if max_rank > min_rank:
            week_df['rank_margin'] = (max_rank - week_df['combined_rank']) / (max_rank - min_rank)
        else:
            week_df['rank_margin'] = 0.5
        
        # === Percentage Method ===
        # 裁判得分百分�?
        judge_sum = week_df['judge_total'].sum()
        week_df['judge_share'] = week_df['judge_total'] / judge_sum if judge_sum > 0 else 0
        # 粉丝份额已有 (fan_vote_share)
        # 综合份额: 越高越安�?
        week_df['combined_share'] = week_df['judge_share'] + week_df['fan_vote_share']
        
        # 归一�? 0 = 最危险(最低combined_share), 1 = 最安全(最高combined_share)
        max_share = week_df['combined_share'].max()
        min_share = week_df['combined_share'].min()
        if max_share > min_share:
            week_df['pct_margin'] = (week_df['combined_share'] - min_share) / (max_share - min_share)
        else:
            week_df['pct_margin'] = 0.5
        
        for _, row in week_df.iterrows():
            results.append({
                'celebrity_name': row['celebrity_name'],
                'week': week,
                'rank_margin': row['rank_margin'],
                'pct_margin': row['pct_margin'],
                'judge_rank': row['judge_rank'],
                'fan_rank': row['fan_rank'],
                'combined_rank': row['combined_rank'],
                'combined_share': row['combined_share'],
                'eliminated': row['eliminated_this_week'],
            })
    
    return pd.DataFrame(results)


def plot_counterfactual_survival():
    df = load_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = {'Rank': '#2E86AB', 'Pct': '#E94F37'}  # 蓝色=Rank, 红色=Pct
    
    for idx, (name, info) in enumerate(CONTROVERSIAL.items()):
        ax = axes[idx]
        season = info['season']
        
        # 计算该赛季的周指�?
        metrics_df = compute_weekly_metrics(df, season)
        
        # 筛选该选手的数�?
        celeb_df = metrics_df[metrics_df['celebrity_name'] == name].copy()
        
        if celeb_df.empty:
            ax.text(0.5, 0.5, f'No data for {name}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} (Season {season})')
            continue
        
        weeks = celeb_df['week'].values
        rank_margins = celeb_df['rank_margin'].values
        pct_margins = celeb_df['pct_margin'].values
        
        # 绘制两条轨迹
        ax.plot(weeks, rank_margins, 'o-', color=colors['Rank'], 
                label='Rank Method', markersize=5, linewidth=1.5)
        ax.plot(weeks, pct_margins, 's--', color=colors['Pct'], 
                label='Percentage Method', markersize=5, linewidth=1.5)
        
        # 找出分歧最大的�?(Point of Divergence)
        diffs = np.abs(rank_margins - pct_margins)
        if len(diffs) > 0:
            max_diff_idx = np.argmax(diffs)
            max_diff_week = weeks[max_diff_idx]
            max_diff = diffs[max_diff_idx]
            
            if max_diff > 0.1:  # 只有差异足够大时才标�?
                ax.axvline(x=max_diff_week, color='gray', linestyle=':', alpha=0.7)
                ax.annotate(f'Max Δ = {max_diff:.2f}', 
                           xy=(max_diff_week, 0.95), 
                           fontsize=8, ha='center', color='gray')
        
        # 危险区域 (Bottom-2 threshold)
        ax.axhspan(0, 0.2, alpha=0.15, color='red', label='Danger Zone')
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlim(min(weeks) - 0.5, max(weeks) + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Week')
        ax.set_ylabel('Normalized Safety Margin')
        ax.set_title(f'{name} (Season {season})', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Counterfactual Survival Trajectories\n(0 = Elimination Line, 1 = Safest)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('d:/2026-repo/figures/counterfactual_survival.png', dpi=800, bbox_inches='tight')
    plt.savefig('d:/2026-repo/figures/counterfactual_survival.pdf', bbox_inches='tight')
    print("Saved: figures/counterfactual_survival.png/pdf")
    plt.show()


if __name__ == '__main__':
    plot_counterfactual_survival()
