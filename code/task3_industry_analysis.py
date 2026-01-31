# -*- coding: utf-8 -*-
"""
Task 3 Extended: Industry/Profession Descriptive Analysis
==========================================================
方案A：Industry作为描述性分析，不放入回归模型

分析内容：
1. 各行业选手分布
2. 各行业表现统计（平均名次、评委分、粉丝投票）
3. 可视化：柱状图、箱线图
4. 无回归分析 - 保持主模型简洁
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def simplify_industry(ind):
    """将细分行业合并为大类"""
    if pd.isna(ind):
        return 'Other'
    ind = ind.strip()
    if ind in ['Actor/Actress']:
        return 'Actor/Actress'
    elif ind in ['Athlete', 'Racing Driver']:
        return 'Athlete'
    elif ind in ['TV Personality', 'News Anchor', 'Sports Broadcaster', 'Radio Personality']:
        return 'TV/Media'
    elif ind in ['Singer/Rapper', 'Musician']:
        return 'Singer/Musician'
    elif ind in ['Model', 'Beauty Pagent']:
        return 'Model'
    elif ind in ['Comedian']:
        return 'Comedian'
    else:
        return 'Other'


def compute_avg_judge(row):
    """计算选手平均每周评委总分"""
    weekly_totals = []
    for w in range(1, 12):
        week_scores = []
        for j in range(1, 5):
            col = f'week{w}_judge{j}_score'
            if col in row.index and pd.notna(row[col]) and row[col] > 0:
                week_scores.append(row[col])
        if week_scores:
            weekly_totals.append(sum(week_scores))
    return np.mean(weekly_totals) if weekly_totals else 0


def main():
    print("=" * 70)
    print("Task 3 Extended: Industry Descriptive Analysis (方案A)")
    print("=" * 70)
    
    # =========================================================================
    # 1. 数据加载
    # =========================================================================
    print("\n[1/5] Loading data...")
    
    raw_df = pd.read_csv('../data/2026_MCM_Problem_C_Data_Cleaned.csv')
    pop_df = pd.read_csv('../data/2026_MCM_Problem_C_Data_Cleaned添加人气后.csv')
    fan_df = pd.read_csv('../data/fan_vote_results_final.csv')
    
    # 合并数据
    df = pop_df.copy()
    df['celebrity_industry'] = raw_df['celebrity_industry']
    df['industry_group'] = df['celebrity_industry'].apply(simplify_industry)
    df['avg_judge_score'] = df.apply(compute_avg_judge, axis=1)
    
    # 合并粉丝投票数据
    fan_summary = fan_df.groupby(['season', 'celebrity_name']).agg({
        'fan_vote_share': 'mean'
    }).reset_index()
    fan_summary.columns = ['season', 'celebrity_name', 'avg_fan_vote_share']
    df = df.merge(fan_summary, on=['season', 'celebrity_name'], how='left')
    
    # 筛选有效数据
    valid_df = df[df['avg_judge_score'] > 0].dropna(subset=['industry_group', 'placement'])
    print(f"    Valid contestants: {len(valid_df)}")
    
    # =========================================================================
    # 2. 行业分布统计
    # =========================================================================
    print("\n[2/5] Computing industry statistics...")
    
    # 原始行业分布
    raw_industry_counts = raw_df['celebrity_industry'].value_counts()
    print("\n    Original Industry Distribution:")
    for ind, cnt in raw_industry_counts.head(10).items():
        print(f"      {ind}: {cnt}")
    
    # 简化后行业分布
    industry_counts = valid_df['industry_group'].value_counts()
    print("\n    Simplified Industry Distribution:")
    for ind, cnt in industry_counts.items():
        print(f"      {ind}: {cnt}")
    
    # =========================================================================
    # 3. 各行业表现统计
    # =========================================================================
    print("\n[3/5] Computing performance by industry...")
    
    industry_stats = valid_df.groupby('industry_group').agg({
        'placement': ['count', 'mean', 'std', 'min'],
        'avg_judge_score': ['mean', 'std'],
        'avg_fan_vote_share': ['mean', 'std']
    }).reset_index()
    
    # 展平列名
    industry_stats.columns = [
        'Industry', 'N', 'Avg_Placement', 'Std_Placement', 'Best_Placement',
        'Avg_Judge', 'Std_Judge', 'Avg_Fan', 'Std_Fan'
    ]
    
    # 计算冠军数量
    winners = valid_df[valid_df['placement'] == 1].groupby('industry_group').size()
    industry_stats['Winners'] = industry_stats['Industry'].map(winners).fillna(0).astype(int)
    industry_stats['Win_Rate'] = (industry_stats['Winners'] / industry_stats['N'] * 100).round(1)
    
    # 按平均名次排序
    industry_stats = industry_stats.sort_values('Avg_Placement')
    
    print("\n    Industry Performance Ranking (by Avg Placement):")
    print("    " + "-" * 80)
    print(f"    {'Industry':<15} {'N':>5} {'Avg Rank':>10} {'Best':>6} {'Judge':>8} {'Fan':>8} {'Win%':>7}")
    print("    " + "-" * 80)
    for _, row in industry_stats.iterrows():
        print(f"    {row['Industry']:<15} {row['N']:>5.0f} {row['Avg_Placement']:>10.2f} "
              f"{row['Best_Placement']:>6.0f} {row['Avg_Judge']:>8.1f} {row['Avg_Fan']:>8.3f} {row['Win_Rate']:>6.1f}%")
    
    # =========================================================================
    # 4. 可视化
    # =========================================================================
    print("\n[4/5] Generating visualizations...")
    
    # 图1：行业表现柱状图（平均名次）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 子图1：平均名次（数值越小越好）
    colors_placement = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(industry_stats)))
    ax1 = axes[0]
    bars = ax1.barh(industry_stats['Industry'], industry_stats['Avg_Placement'], 
                    color=colors_placement, edgecolor='white', linewidth=0.8)
    for i, (_, row) in enumerate(industry_stats.iterrows()):
        ax1.text(row['Avg_Placement'] + 0.2, i, f"n={row['N']:.0f}", 
                 va='center', fontsize=9, color='gray')
    ax1.set_xlabel('Average Placement (Lower = Better)', fontsize=11)
    ax1.set_title('A) Performance by Industry\n(Avg Final Placement)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(industry_stats['Avg_Placement']) * 1.25)
    
    # 子图2：平均评委分
    ax2 = axes[1]
    colors_judge = plt.cm.Blues(np.linspace(0.4, 0.9, len(industry_stats)))
    sorted_by_judge = industry_stats.sort_values('Avg_Judge', ascending=True)
    ax2.barh(sorted_by_judge['Industry'], sorted_by_judge['Avg_Judge'], 
             color=colors_judge, edgecolor='white', linewidth=0.8)
    ax2.set_xlabel('Average Judge Score (0-40)', fontsize=11)
    ax2.set_title('B) Judge Score by Industry', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 35)
    
    # 子图3：平均粉丝投票
    ax3 = axes[2]
    colors_fan = plt.cm.Oranges(np.linspace(0.4, 0.9, len(industry_stats)))
    sorted_by_fan = industry_stats.sort_values('Avg_Fan', ascending=True)
    ax3.barh(sorted_by_fan['Industry'], sorted_by_fan['Avg_Fan'] * 100, 
             color=colors_fan, edgecolor='white', linewidth=0.8)
    ax3.set_xlabel('Average Fan Vote Share (%)', fontsize=11)
    ax3.set_title('C) Fan Vote by Industry', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/task3_industry_effect.png', dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print("    Saved: task3_industry_effect.png")
    
    # 图2：箱线图比较
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 准备数据（按平均名次排序）
    order = industry_stats['Industry'].tolist()
    
    # 箱线图1：名次分布
    ax1 = axes[0]
    box_data_placement = [valid_df[valid_df['industry_group'] == ind]['placement'].dropna() 
                          for ind in order]
    bp1 = ax1.boxplot(box_data_placement, labels=order, patch_artist=True, vert=False)
    for patch, color in zip(bp1['boxes'], colors_placement):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xlabel('Final Placement', fontsize=11)
    ax1.set_title('Placement Distribution by Industry', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # 箱线图2：评委分分布
    ax2 = axes[1]
    box_data_judge = [valid_df[valid_df['industry_group'] == ind]['avg_judge_score'].dropna() 
                      for ind in order]
    bp2 = ax2.boxplot(box_data_judge, labels=order, patch_artist=True, vert=False)
    for patch, color in zip(bp2['boxes'], colors_placement):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xlabel('Average Judge Score', fontsize=11)
    ax2.set_title('Judge Score Distribution by Industry', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../figures/task3_industry_boxplot.png', dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print("    Saved: task3_industry_boxplot.png")
    
    # =========================================================================
    # 5. 生成报告
    # =========================================================================
    print("\n[5/5] Generating markdown report...")
    
    output_lines = []
    output_lines.append("# Task 3 Extended: Industry Descriptive Analysis")
    output_lines.append("")
    output_lines.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    output_lines.append("## Overview")
    output_lines.append("")
    output_lines.append("本节对选手的职业/行业背景进行**描述性分析**，探索不同行业选手在DWTS中的表现差异。")
    output_lines.append("")
    output_lines.append("> **注意**：Industry变量未纳入主回归模型，以保持模型简洁性。本节仅提供描述性统计和可视化。")
    output_lines.append("")
    
    # 行业分布
    output_lines.append("## 1. Industry Distribution")
    output_lines.append("")
    output_lines.append("### 1.1 Original Categories")
    output_lines.append("")
    output_lines.append("| Industry | Count |")
    output_lines.append("|----------|-------|")
    for ind, cnt in raw_industry_counts.head(12).items():
        output_lines.append(f"| {ind} | {cnt} |")
    output_lines.append("")
    
    output_lines.append("### 1.2 Simplified Categories")
    output_lines.append("")
    output_lines.append("We consolidated the original categories into 7 groups for clearer analysis:")
    output_lines.append("")
    output_lines.append("| Group | Includes | N |")
    output_lines.append("|-------|----------|---|")
    output_lines.append(f"| Actor/Actress | Actor/Actress | {industry_counts.get('Actor/Actress', 0)} |")
    output_lines.append(f"| Athlete | Athlete, Racing Driver | {industry_counts.get('Athlete', 0)} |")
    output_lines.append(f"| TV/Media | TV Personality, News Anchor, Broadcaster | {industry_counts.get('TV/Media', 0)} |")
    output_lines.append(f"| Singer/Musician | Singer/Rapper, Musician | {industry_counts.get('Singer/Musician', 0)} |")
    output_lines.append(f"| Model | Model, Beauty Pageant | {industry_counts.get('Model', 0)} |")
    output_lines.append(f"| Comedian | Comedian | {industry_counts.get('Comedian', 0)} |")
    output_lines.append(f"| Other | Socialite, Entrepreneur, etc. | {industry_counts.get('Other', 0)} |")
    output_lines.append("")
    
    # 表现统计
    output_lines.append("## 2. Performance by Industry")
    output_lines.append("")
    output_lines.append("### 2.1 Summary Statistics")
    output_lines.append("")
    output_lines.append("| Industry | N | Avg Placement | Best | Avg Judge | Avg Fan Vote | Winners | Win Rate |")
    output_lines.append("|----------|---|---------------|------|-----------|--------------|---------|----------|")
    for _, row in industry_stats.iterrows():
        output_lines.append(
            f"| {row['Industry']} | {row['N']:.0f} | {row['Avg_Placement']:.2f} | "
            f"{row['Best_Placement']:.0f} | {row['Avg_Judge']:.1f} | {row['Avg_Fan']:.3f} | "
            f"{row['Winners']:.0f} | {row['Win_Rate']:.1f}% |"
        )
    output_lines.append("")
    
    # 发现
    best = industry_stats.loc[industry_stats['Avg_Placement'].idxmin()]
    worst = industry_stats.loc[industry_stats['Avg_Placement'].idxmax()]
    most_winners = industry_stats.loc[industry_stats['Winners'].idxmax()]
    highest_judge = industry_stats.loc[industry_stats['Avg_Judge'].idxmax()]
    highest_fan = industry_stats.loc[industry_stats['Avg_Fan'].idxmax()]
    
    output_lines.append("### 2.2 Key Observations")
    output_lines.append("")
    output_lines.append(f"1. **Best Average Placement**: {best['Industry']} (avg rank: {best['Avg_Placement']:.2f})")
    output_lines.append(f"2. **Worst Average Placement**: {worst['Industry']} (avg rank: {worst['Avg_Placement']:.2f})")
    output_lines.append(f"3. **Most Winners**: {most_winners['Industry']} ({most_winners['Winners']:.0f} wins)")
    output_lines.append(f"4. **Highest Judge Scores**: {highest_judge['Industry']} (avg: {highest_judge['Avg_Judge']:.1f})")
    output_lines.append(f"5. **Highest Fan Vote Share**: {highest_fan['Industry']} (avg: {highest_fan['Avg_Fan']:.3f})")
    output_lines.append("")
    
    # 图表
    output_lines.append("## 3. Visualizations")
    output_lines.append("")
    output_lines.append("### Figure: Industry Effect Comparison")
    output_lines.append("")
    output_lines.append("![Industry Effect](../figures/task3_industry_effect.png)")
    output_lines.append("")
    output_lines.append("### Figure: Industry Distribution (Boxplot)")
    output_lines.append("")
    output_lines.append("![Industry Boxplot](../figures/task3_industry_boxplot.png)")
    output_lines.append("")
    
    # 讨论
    output_lines.append("## 4. Discussion")
    output_lines.append("")
    output_lines.append("### Why Not Include Industry in the Regression Model?")
    output_lines.append("")
    output_lines.append("1. **Parsimony**: Including 6+ dummy variables would reduce model interpretability.")
    output_lines.append("2. **Collinearity Concerns**: Industry may correlate with age and popularity.")
    output_lines.append("3. **Sample Size**: Some categories (e.g., Comedian) have limited observations.")
    output_lines.append("4. **Focus**: The main model emphasizes universal factors (age, popularity, partner effects).")
    output_lines.append("")
    output_lines.append("### Interpretation Notes")
    output_lines.append("")
    output_lines.append("- **Athletes** often perform well, possibly due to physical conditioning and competitive mindset.")
    output_lines.append("- **Actors/Actresses** show strong performance, potentially from stage experience and camera comfort.")
    output_lines.append("- **Singers/Musicians** may have inherent rhythm advantages but mixed results overall.")
    output_lines.append("- Industry effects are **confounded** with other factors and should not be interpreted causally.")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    output_lines.append("*This analysis is supplementary to the main regression results in Task 3.*")
    
    # 保存报告
    output_file = '../data/task3_industry_analysis.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"    Saved: {output_file}")
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("Industry Descriptive Analysis Complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Report: data/task3_industry_analysis.md")
    print(f"  - Figure: figures/task3_industry_effect.png")
    print(f"  - Figure: figures/task3_industry_boxplot.png")
    print("")


if __name__ == "__main__":
    main()
