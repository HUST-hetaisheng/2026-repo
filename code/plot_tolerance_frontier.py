# -*- coding: utf-8 -*-
"""
Tolerance Frontier Map
Compare Rank vs Percentage method tolerance for fan favorites

X-axis: Judge Rank (1=Best, N=Worst)
Y-axis: Fan Rank (1=Best, N=Worst)
Background: Safe zone (green) vs Danger zone (red)
Data points: Controversial contestants weekly positions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


CONTROVERSIAL = ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]


def load_data():
    df = pd.read_csv("d:/2026-repo/data/fan_vote_results_final.csv")
    return df


def compute_all_weekly_ranks(df):
    results = []
    
    for season in df['season'].unique():
        season_df = df[df['season'] == season].copy()
        
        for week in season_df['week'].unique():
            week_df = season_df[season_df['week'] == week].copy()
            week_df = week_df[week_df['judge_total'] > 0].copy()
            
            if len(week_df) < 2:
                continue
            
            n = len(week_df)
            
            # Rank Method
            week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False, method='average')
            week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False, method='average')
            week_df['combined_rank'] = week_df['judge_rank'] + week_df['fan_rank']
            
            # Percentage Method
            judge_sum = week_df['judge_total'].sum()
            week_df['judge_share'] = week_df['judge_total'] / judge_sum
            week_df['combined_share'] = week_df['judge_share'] + week_df['fan_vote_share']
            week_df['percent_rank'] = week_df['combined_share'].rank(ascending=False, method='average')
            
            week_df['judge_rank_norm'] = (week_df['judge_rank'] - 1) / (n - 1) if n > 1 else 0.5
            week_df['fan_rank_norm'] = (week_df['fan_rank'] - 1) / (n - 1) if n > 1 else 0.5
            
            for _, row in week_df.iterrows():
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': row['celebrity_name'],
                    'n_contestants': n,
                    'judge_rank': row['judge_rank'],
                    'fan_rank': row['fan_rank'],
                    'judge_rank_norm': row['judge_rank_norm'],
                    'fan_rank_norm': row['fan_rank_norm'],
                    'combined_rank': row['combined_rank'],
                    'combined_share': row['combined_share'],
                    'percent_rank': row['percent_rank'],
                    'judge_share': row['judge_share'],
                    'fan_vote_share': row['fan_vote_share'],
                    'eliminated': row['eliminated_this_week'],
                    'is_controversial': row['celebrity_name'] in CONTROVERSIAL,
                })
    
    return pd.DataFrame(results)


def compute_percent_frontier_slope(df):
    judge_share_std = df['judge_share'].std()
    fan_share_std = df['fan_vote_share'].std()
    variance_ratio = fan_share_std / judge_share_std if judge_share_std > 0 else 1.0
    
    print(f"Judge Share Std: {judge_share_std:.4f}")
    print(f"Fan Share Std: {fan_share_std:.4f}")
    print(f"Variance Ratio (Fan/Judge): {variance_ratio:.2f}")
    
    return variance_ratio


def plot_tolerance_frontier():
    df = load_data()
    all_data = compute_all_weekly_ranks(df)
    
    variance_ratio = compute_percent_frontier_slope(all_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    N = 10
    x = np.linspace(1, N, 100)
    y = np.linspace(1, N, 100)
    X, Y = np.meshgrid(x, y)
    
    threshold_rank = 2 * (N - 1)
    
    # Left: Rank Method
    ax1 = axes[0]
    Z_rank = X + Y
    
    cmap_rg = LinearSegmentedColormap.from_list('safety', ['#2ecc71', '#f1c40f', '#e74c3c'])
    im1 = ax1.contourf(X, Y, Z_rank, levels=20, cmap=cmap_rg, alpha=0.6)
    
    y_boundary = threshold_rank - x
    ax1.plot(x, y_boundary, 'k-', linewidth=2.5, label=f'Elimination Boundary\n(slope = -1)')
    ax1.fill_between(x, y_boundary, N, alpha=0.3, color='red')
    
    ax1.annotate('SAFE ZONE', xy=(2, 2), fontsize=12, fontweight='bold', color='darkgreen')
    ax1.annotate('DANGER ZONE', xy=(7, 8), fontsize=12, fontweight='bold', color='darkred')
    
    ax1.set_xlim(1, N)
    ax1.set_ylim(1, N)
    ax1.set_xlabel('Judge Rank (1=Best, N=Worst)', fontsize=11)
    ax1.set_ylabel('Fan Rank (1=Best, N=Worst)', fontsize=11)
    ax1.set_title('Rank Method\n$R_{judge} + R_{fan} = Threshold$', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Percentage Method
    ax2 = axes[1]
    
    effective_slope = -1 / variance_ratio
    Z_percent = X + Y / variance_ratio
    
    im2 = ax2.contourf(X, Y, Z_percent, levels=20, cmap=cmap_rg, alpha=0.6)
    
    y_boundary_pct = (threshold_rank - x) * variance_ratio
    y_boundary_pct = np.clip(y_boundary_pct, 1, N)
    
    ax2.plot(x, y_boundary_pct, 'k-', linewidth=2.5, 
             label=f'Elimination Boundary\n(slope = {-variance_ratio:.1f})')
    ax2.fill_between(x, y_boundary_pct, N, alpha=0.3, color='red', 
                     where=(y_boundary_pct < N))
    
    ax2.annotate('SAFE ZONE\n(Expanded)', xy=(2, 2), fontsize=12, fontweight='bold', color='darkgreen')
    ax2.annotate('DANGER\nZONE', xy=(8, 9), fontsize=10, fontweight='bold', color='darkred')
    
    ax2.set_xlim(1, N)
    ax2.set_ylim(1, N)
    ax2.set_xlabel('Judge Rank (1=Best, N=Worst)', fontsize=11)
    ax2.set_ylabel('Fan Rank (1=Best, N=Worst)', fontsize=11)
    ax2.set_title('Percentage Method\n(Fan vote variance amplifies tolerance)', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot controversial contestants
    controversial_data = all_data[all_data['is_controversial']].copy()
    
    marker_style = {'Jerry Rice': 'o', 'Billy Ray Cyrus': 's', 
                   'Bristol Palin': '^', 'Bobby Bones': 'D'}
    color_map = {'Jerry Rice': '#3498db', 'Billy Ray Cyrus': '#9b59b6',
                'Bristol Palin': '#e67e22', 'Bobby Bones': '#1abc9c'}
    
    for _, row in controversial_data.iterrows():
        n = row['n_contestants']
        jr = row['judge_rank']
        fr = row['fan_rank']
        
        jr_scaled = 1 + (jr - 1) / (n - 1) * (N - 1) if n > 1 else N/2
        fr_scaled = 1 + (fr - 1) / (n - 1) * (N - 1) if n > 1 else N/2
        
        name = row['celebrity_name']
        ax1.scatter(jr_scaled, fr_scaled, marker=marker_style.get(name, 'o'),
                   c=color_map.get(name, 'blue'), s=60, edgecolors='white', 
                   linewidths=0.5, alpha=0.8, zorder=5)
        ax2.scatter(jr_scaled, fr_scaled, marker=marker_style.get(name, 'o'),
                   c=color_map.get(name, 'blue'), s=60, edgecolors='white',
                   linewidths=0.5, alpha=0.8, zorder=5)
    
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', label='Jerry Rice'),
        mpatches.Patch(facecolor='#9b59b6', label='Billy Ray Cyrus'),
        mpatches.Patch(facecolor='#e67e22', label='Bristol Palin'),
        mpatches.Patch(facecolor='#1abc9c', label='Bobby Bones'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               fontsize=10, title='Controversial Contestants', 
               bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Tolerance Frontier Map: Structural Bias Visualization\n'
                 '(Why Percentage Method is More "Forgiving" to Fan Favorites)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig('d:/2026-repo/figures/tolerance_frontier.png', dpi=1000, bbox_inches='tight')
    plt.savefig('d:/2026-repo/figures/tolerance_frontier.pdf', bbox_inches='tight')
    print("Saved: figures/tolerance_frontier.png/pdf")
    plt.show()


if __name__ == '__main__':
    plot_tolerance_frontier()
