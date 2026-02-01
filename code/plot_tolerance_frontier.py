"""
Tolerance Frontier Map (规则容忍度边界图)
展示 Rank 制 vs Percentage 制 对"偏科生"的容忍极限

核心思想:
- Rank制: R_judge + R_fan = threshold (直线, 斜率 -1)
- Percent制: 由于粉丝票方差大, 粉丝Rank每提升1名能抵消更多裁判Rank

X轴: Judge Rank (1=Best, N=Worst)
Y轴: Fan Rank (1=Best, N=Worst)
背景: 安全区(绿) vs 淘汰区(红)
数据点: 争议选手的每周位置
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


# === Controversial contestants ===
CONTROVERSIAL = ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]


def load_data():
    """加载fan_vote_results_final.csv"""
    df = pd.read_csv("d:/2026-repo/data/fan_vote_results_final.csv")
    return df


def compute_all_weekly_ranks(df):
    """
    计算所有赛季所有周的排名数据
    返回: DataFrame with judge_rank, fan_rank, and combined scores
    """
    results = []
    
    for season in df['season'].unique():
        season_df = df[df['season'] == season].copy()
        
        for week in season_df['week'].unique():
            week_df = season_df[season_df['week'] == week].copy()
            
            # 只保留有效选手
            week_df = week_df[week_df['judge_total'] > 0].copy()
            
            if len(week_df) < 2:
                continue
            
            n = len(week_df)
            
            # === Rank Method ===
            week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False, method='average')
            week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False, method='average')
            week_df['combined_rank'] = week_df['judge_rank'] + week_df['fan_rank']
            
            # === Percentage Method ===
            judge_sum = week_df['judge_total'].sum()
            week_df['judge_share'] = week_df['judge_total'] / judge_sum
            week_df['combined_share'] = week_df['judge_share'] + week_df['fan_vote_share']
            
            # 计算 Percent 制下的等效排名
            week_df['percent_rank'] = week_df['combined_share'].rank(ascending=False, method='average')
            
            # 计算归一化坐标 (1 to N -> 0 to 1, 其中 0=best, 1=worst)
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
    """
    计算 Percent 制下的边界斜率
    
    关键问题: 在 Percent 制下, 粉丝份额的方差远大于裁判份额
    因此粉丝Rank每提升1名, 能抵消的裁判Rank可能是 2-3 名
    
    方法: 用线性回归估计 judge_share vs fan_vote_share 的方差比
    """
    # 计算方差比
    judge_share_std = df['judge_share'].std()
    fan_share_std = df['fan_vote_share'].std()
    
    # 粉丝份额的方差通常更大, 意味着粉丝Rank变化1单位的份额变化更大
    # 因此在 Percent 制下, 粉丝票的"权重"被隐形放大
    variance_ratio = fan_share_std / judge_share_std if judge_share_std > 0 else 1.0
    
    print(f"Judge Share Std: {judge_share_std:.4f}")
    print(f"Fan Share Std: {fan_share_std:.4f}")
    print(f"Variance Ratio (Fan/Judge): {variance_ratio:.2f}")
    
    # 边界斜率: 在 Rank 制下是 -1 (直线)
    # 在 Percent 制下, 由于粉丝方差大, 斜率更陡峭 (约 -variance_ratio)
    return variance_ratio


def plot_tolerance_frontier():
    df = load_data()
    all_data = compute_all_weekly_ranks(df)
    
    # 计算方差比
    variance_ratio = compute_percent_frontier_slope(all_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 网格设置
    N = 10  # 假设典型赛季有10个选手
    x = np.linspace(1, N, 100)  # Judge Rank
    y = np.linspace(1, N, 100)  # Fan Rank
    X, Y = np.meshgrid(x, y)
    
    # 淘汰阈值 (假设 Bottom-2, 即最后2名会进入淘汰区)
    # Rank制: combined_rank > 2*(N-1) 意味着危险
    threshold_rank = 2 * (N - 1)  # 边界线
    
    # ==================== 左图: Rank 制 ====================
    ax1 = axes[0]
    
    # Rank 制下的 combined score = judge_rank + fan_rank
    Z_rank = X + Y
    
    # 背景填充: 绿色(安全) -> 红色(危险)
    cmap_rg = LinearSegmentedColormap.from_list('safety', ['#2ecc71', '#f1c40f', '#e74c3c'])
    im1 = ax1.contourf(X, Y, Z_rank, levels=20, cmap=cmap_rg, alpha=0.6)
    
    # 画淘汰边界线 (斜率 = -1)
    # X + Y = threshold
    y_boundary = threshold_rank - x
    ax1.plot(x, y_boundary, 'k-', linewidth=2.5, label=f'Elimination Boundary\n(slope = -1)')
    ax1.fill_between(x, y_boundary, N, alpha=0.3, color='red')
    
    # 添加标注
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
    
    # ==================== 右图: Percentage 制 ====================
    ax2 = axes[1]
    
    # Percent 制下, 粉丝票方差大, 等效为粉丝Rank权重更高
    # 边界线不再是斜率 -1, 而是更陡峭
    # 设 effective_slope = -1/variance_ratio (粉丝1名 ≈ 裁判 variance_ratio 名)
    effective_slope = -1 / variance_ratio
    
    # 调整后的 combined score (模拟 Percent 制的效果)
    Z_percent = X + Y / variance_ratio  # 粉丝Rank的影响被"缩小"
    
    im2 = ax2.contourf(X, Y, Z_percent, levels=20, cmap=cmap_rg, alpha=0.6)
    
    # 画淘汰边界线 (斜率更陡峭)
    # X + Y/variance_ratio = threshold / something
    # 为了对比, 用同样的"排名阈值"
    threshold_percent = threshold_rank / (1 + 1/variance_ratio) * (1 + 1/variance_ratio)
    y_boundary_pct = (threshold_rank - x) * variance_ratio
    y_boundary_pct = np.clip(y_boundary_pct, 1, N)
    
    ax2.plot(x, y_boundary_pct, 'k-', linewidth=2.5, 
             label=f'Elimination Boundary\n(slope ≈ {-variance_ratio:.1f})')
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
    
    # ==================== 叠加争议选手的数据点 ====================
    controversial_data = all_data[all_data['is_controversial']].copy()
    
    # 归一化到 1-N 范围
    for _, row in controversial_data.iterrows():
        n = row['n_contestants']
        jr = row['judge_rank']
        fr = row['fan_rank']
        
        # 缩放到 1-10 范围
        jr_scaled = 1 + (jr - 1) / (n - 1) * (N - 1) if n > 1 else N/2
        fr_scaled = 1 + (fr - 1) / (n - 1) * (N - 1) if n > 1 else N/2
        
        # 在两个图上都画点
        marker_style = {'Jerry Rice': 'o', 'Billy Ray Cyrus': 's', 
                       'Bristol Palin': '^', 'Bobby Bones': 'D'}
        color_map = {'Jerry Rice': '#3498db', 'Billy Ray Cyrus': '#9b59b6',
                    'Bristol Palin': '#e67e22', 'Bobby Bones': '#1abc9c'}
        
        name = row['celebrity_name']
        ax1.scatter(jr_scaled, fr_scaled, marker=marker_style.get(name, 'o'),
                   c=color_map.get(name, 'blue'), s=60, edgecolors='white', 
                   linewidths=0.5, alpha=0.8, zorder=5)
        ax2.scatter(jr_scaled, fr_scaled, marker=marker_style.get(name, 'o'),
                   c=color_map.get(name, 'blue'), s=60, edgecolors='white',
                   linewidths=0.5, alpha=0.8, zorder=5)
    
    # 添加图例
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
    plt.savefig('d:/2026-repo/figures/tolerance_frontier.png', dpi=300, bbox_inches='tight')
    plt.savefig('d:/2026-repo/figures/tolerance_frontier.pdf', bbox_inches='tight')
    print("Saved: figures/tolerance_frontier.png/pdf")
    plt.show()


if __name__ == '__main__':
    plot_tolerance_frontier()
