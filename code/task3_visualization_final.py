"""
Task 3 Final Visualizations for MCM 2026
=========================================
Generates publication-ready figures for Task 3 analysis.

Author: Team
Date: 2025-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

# ============================================================
# 自定义配色方案 (Custom Color Palette)
# ============================================================
# 渐变序列：从深红到深蓝
GRADIENT_COLORS = [
    '#831A21',  # 深红 (负值最大)
    '#A13D3B',
    '#C16D58',
    '#ECD0B4',
    '#F2EBE5',  # 中性色
    '#E8EDF1',
    '#C8D6E7',
    '#9EBCDB',
    '#7091C7',
    '#4E70AF',
    '#375093',  # 深蓝 (正值最大)
]

# 创建自定义 colormap（用于热力图）
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom_diverging', GRADIENT_COLORS, N=256)
CUSTOM_CMAP_R = LinearSegmentedColormap.from_list('custom_diverging_r', GRADIENT_COLORS[::-1], N=256)

# 主题色
COLOR_PRIMARY = '#375093'      # 深蓝
COLOR_SECONDARY = '#7091C7'    # 中蓝
COLOR_ACCENT = '#831A21'       # 深红
COLOR_WARM = '#C16D58'         # 暖橙
COLOR_NEUTRAL = '#E8EDF1'      # 浅灰蓝
COLOR_BG = '#F2EBE5'           # 米色背景

# 对比色组（用于分组条形图）
COLOR_JUDGE = '#375093'        # 深蓝 - 评委
COLOR_FAN = '#831A21'          # 深红 - 粉丝
COLOR_PLACEMENT = '#4E70AF'    # 中蓝 - 排名

# 渐变色组（用于排名条形图）
def get_gradient_colors(n, reverse=False):
    """生成 n 个渐变色"""
    indices = np.linspace(0, len(GRADIENT_COLORS)-1, n).astype(int)
    colors = [GRADIENT_COLORS[i] for i in indices]
    return colors[::-1] if reverse else colors

# ============================================================
# 全局样式设置
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体与 Times 兼容
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.facecolor'] = '#FAFAFA'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# ==== 图1: 系数热力图 ====
def plot_coefficient_heatmap():
    """Generate coefficient comparison heatmap with custom colormap"""
    df = pd.read_csv('../data/task3_regression_coefficients.csv')
    
    # Filter to contestant-level models only
    df = df[df['Model'].isin(['Judge_Score', 'Fan_Vote', 'Placement'])]
    
    # Select key variables (exclude intercept)
    key_vars = ['age', 'celeb_popularity', 'partner_popularity', 'bmi', 
                'dance_experience_score', 'is_us',
                'Industry_[T.Actor]', 'Industry_[T.Athlete]', 'Industry_[T.Singer]',
                'Industry_[T.SocialMedia]', 'Industry_[T.Model]']
    
    df = df[df['Variable'].isin(key_vars)]
    
    # Pivot to matrix
    coef_matrix = df.pivot(index='Variable', columns='Model', values='Coefficient')
    pval_matrix = df.pivot(index='Variable', columns='Model', values='p_value')
    
    # Rename for display
    var_names = {
        'age': 'Age',
        'celeb_popularity': 'Celebrity Popularity',
        'partner_popularity': 'Partner Popularity',
        'bmi': 'BMI',
        'dance_experience_score': 'Dance Experience',
        'is_us': 'US Nationality',
        'Industry_[T.Actor]': 'Industry: Actor',
        'Industry_[T.Athlete]': 'Industry: Athlete',
        'Industry_[T.Singer]': 'Industry: Singer',
        'Industry_[T.SocialMedia]': 'Industry: Social Media',
        'Industry_[T.Model]': 'Industry: Model'
    }
    coef_matrix = coef_matrix.rename(index=var_names)
    pval_matrix = pval_matrix.rename(index=var_names)
    
    # Column order
    col_order = ['Judge_Score', 'Fan_Vote', 'Placement']
    coef_matrix = coef_matrix[col_order]
    pval_matrix = pval_matrix[col_order]
    
    # Create significance annotation
    def sig_stars(p):
        if pd.isna(p): return ''
        if p < 0.01: return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        else: return ''
    
    annot = pval_matrix.map(sig_stars)
    
    # Normalize coefficients for comparison (Z-score within each column)
    coef_norm = coef_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    # Plot with custom styling
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用自定义配色
    sns.heatmap(coef_norm, annot=annot, fmt='', cmap=CUSTOM_CMAP_R, center=0,
                linewidths=1.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Standardized Coefficient', 'shrink': 0.8},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    # 美化标题和标签
    ax.set_title('Regression Coefficients Comparison', 
                 fontweight='bold', fontsize=16, pad=20, color='#2C3E50')
    ax.set_xlabel('Dependent Variable', fontsize=13, labelpad=10, color='#34495E')
    ax.set_ylabel('Independent Variable', fontsize=13, labelpad=10, color='#34495E')
    
    # Rename column labels with better formatting
    ax.set_xticklabels(['Judge Score', 'Fan Vote', 'Placement'], 
                       fontsize=12, fontweight='medium', color='#2C3E50')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, color='#34495E')
    
    # 调整 colorbar 样式
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Standardized Coefficient', fontsize=11, color='#34495E')
    
    plt.tight_layout()
    plt.savefig('../figures/task3_2_coefficient_heatmap.png', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("[VIZ] Saved: task3_2_coefficient_heatmap.png")


# ==== 图1b: 自变量相关性热力图 ====
def plot_variable_correlation():
    """Generate correlation heatmap for independent variables"""
    df = pd.read_csv('../data/task3_contestant_summary_v3.csv')
    
    # 选择数值型自变量
    numeric_vars = ['age', 'celeb_popularity', 'partner_popularity', 'bmi', 
                    'dance_experience_score', 'is_us', 
                    'avg_social_media_popularity', 'avg_google_search_volume']
    
    # 筛选存在的列
    available_vars = [v for v in numeric_vars if v in df.columns]
    df_numeric = df[available_vars].dropna()
    
    # 计算相关系数矩阵
    corr_matrix = df_numeric.corr()
    
    # 变量名重命名（美化显示）
    var_labels = {
        'age': 'Age',
        'celeb_popularity': 'Celebrity\nPopularity',
        'partner_popularity': 'Partner\nPopularity',
        'bmi': 'BMI',
        'dance_experience_score': 'Dance\nExperience',
        'is_us': 'US\nNationality',
        'avg_social_media_popularity': 'Social Media\nPopularity',
        'avg_google_search_volume': 'Google\nSearch'
    }
    
    corr_matrix = corr_matrix.rename(index=var_labels, columns=var_labels)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    # 创建掩码（只显示下三角）
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # 使用自定义配色
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=CUSTOM_CMAP_R, 
                center=0, vmin=-1, vmax=1,
                mask=mask,
                linewidths=1, linecolor='white', ax=ax,
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
                annot_kws={'fontsize': 11, 'fontweight': 'medium'})
    
    ax.set_title('Correlation Matrix of Independent Variables', 
                 fontweight='bold', fontsize=16, pad=20, color='#2C3E50')
    
    # 调整标签
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, color='#34495E', rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, color='#34495E', rotation=0)
    
    # 调整 colorbar 样式
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Pearson Correlation', fontsize=12, color='#34495E')
    
    plt.tight_layout()
    plt.savefig('../figures/task3_2_variable_correlation.png', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("[VIZ] Saved: task3_2_variable_correlation.png")


# ==== 图2: ICC 方差分解 ====
def plot_icc_variance():
    """Plot ICC as elegant donut charts with custom colors"""
    icc_data = pd.read_csv('../data/task3_icc_results.csv')
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # 每个模型使用不同的主色调
    model_colors = [
        (COLOR_JUDGE, '#C8D6E7'),      # Judge: 深蓝 + 浅蓝
        (COLOR_FAN, '#ECD0B4'),         # Fan: 深红 + 暖米
        (COLOR_PLACEMENT, '#E8EDF1'),   # Placement: 中蓝 + 浅灰
    ]
    
    for idx, (_, row) in enumerate(icc_data.iterrows()):
        ax = axes[idx]
        
        partner_pct = row['ICC'] * 100
        residual_pct = 100 - partner_pct
        
        colors = [model_colors[idx][0], model_colors[idx][1]]
        
        # Create elegant donut chart
        wedges, texts, autotexts = ax.pie(
            [partner_pct, residual_pct],
            autopct=lambda pct: f'{pct:.1f}%' if pct > 10 else '',
            colors=colors,
            explode=(0.03, 0),
            startangle=90,
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
            textprops={'fontsize': 12, 'color': '#2C3E50'}
        )
        
        # 在中心添加标签
        ax.text(0, 0, f'{partner_pct:.1f}%', ha='center', va='center', 
                fontsize=18, fontweight='bold', color=model_colors[idx][0])
        
        # 添加图例标签（在下方）
        ax.text(0, -1.4, 'Partner Effect', ha='center', va='center', 
                fontsize=10, color=model_colors[idx][0], fontweight='medium')
        
        # 模型标题
        model_titles = ['Judge Score', 'Fan Vote', 'Placement']
        ax.set_title(model_titles[idx], fontweight='bold', fontsize=14, 
                     pad=15, color='#2C3E50')
    
    # 总标题
    fig.suptitle('Variance Explained by Professional Dancer (ICC)', 
                 fontsize=16, fontweight='bold', y=1.02, color='#2C3E50')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_PRIMARY, edgecolor='white', label='Partner Effect'),
        Patch(facecolor='#E8EDF1', edgecolor='white', label='Other Factors')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig('../figures/task3_2_icc_variance.png', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("[VIZ] Saved: task3_2_icc_variance.png")


# ==== 图3: 舞伴排名 Top 10 ====
def plot_partner_top10():
    """Plot top 10 professional dancers with gradient colors and champion effect annotation"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'task3_partner_ranking_v3.csv'))
    df = df.head(10)  # Top 10
    
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor('white')
    
    y_pos = np.arange(len(df))
    
    # 蓝色渐变系列（第一名用深红色突出）
    blue_gradient = ['#375093', '#4E70AF', '#5A7DB8', '#6A8BC1', '#7A99CA', 
                     '#8AA7D3', '#9EBCDB', '#AEC9E3', '#BED6EB', '#C8D6E7']
    # 第一名使用强调色
    #bar_colors = [COLOR_ACCENT] + blue_gradient[1:]
    bar_colors = blue_gradient
    bars = ax.barh(y_pos, df['Avg_Placement'], color=bar_colors, 
                   edgecolor='white', height=0.72, linewidth=1.5)
    
    # Add annotations with better styling
    for i, (_, row) in enumerate(df.iterrows()):
        # 出场次数标签（加粗）
        ax.text(row['Avg_Placement'] + 0.12, i, f"n = {int(row['Appearances'])}", 
                va='center', fontsize=10, color='#7F8C8D', fontweight='bold')
        # 精确数值（在条形内部）
        if row['Avg_Placement'] > 1.5:
            ax.text(row['Avg_Placement'] - 0.15, i, f"{row['Avg_Placement']:.2f}", 
                    va='center', ha='right', fontsize=10, color='white', fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Partner'], fontsize=12, fontweight='bold', color='#2C3E50')
    ax.invert_yaxis()  # Best at top
    ax.set_xlabel('Average Placement (Lower = Better)', fontsize=13, 
                  labelpad=10, color='#34495E')
    ax.set_title('Top 10 Professional Dancers by Performance', 
                 fontsize=16, fontweight='bold', pad=15, color='#2C3E50')
    
    # Highlight Derek Hough (第一名) - 冠军效应标注
    ax.get_yticklabels()[0].set_color(COLOR_ACCENT)
    
    # 添加"Champion Effect"标注框（放在bar中间，无箭头，保持原样式）
    first_placement = df['Avg_Placement'].iloc[0]
    ax.text(first_placement / 2, 0, 'Champion\nEffect', 
            fontsize=11, fontweight='bold', color=COLOR_ACCENT,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDF2F2', 
                      edgecolor=COLOR_ACCENT, linewidth=1.5))
    
    ax.set_xlim(0, 8)
    ax.set_ylim(9.8, -0.6)  # 调整上方空间
    
    # 均值参考线
    mean_placement = df['Avg_Placement'].mean()
    ax.axvline(x=mean_placement, color="#F61E1E", linestyle='--', 
               alpha=0.6, linewidth=1.5)
    ax.text(mean_placement + 0.08, 9.5, f'Mean: {mean_placement:.2f}', 
            color='#F61E1E', fontsize=10, style='italic')
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'task3_2_partner_ranking_top10.png'), bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("[VIZ] Saved: task3_2_partner_ranking_top10.png")


# ==== 图3-2: 舞伴排名 Top 10 双轴图（加 Avg_Judge 折线）====
def plot_partner_top10_2():
    """Plot top 10 professional dancers with dual-axis: Avg_Placement bars + Avg_Judge line"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'task3_partner_ranking_v3.csv'))
    df = df.head(10)  # Top 10
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    y_pos = np.arange(len(df))
    
    # 蓝色渐变系列
    blue_gradient = ['#375093', '#4E70AF', '#5A7DB8', '#6A8BC1', '#7A99CA', 
                     '#8AA7D3', '#9EBCDB', '#AEC9E3', '#BED6EB', '#C8D6E7']
    bar_colors = blue_gradient
    
    # 左轴：Avg_Placement 条形图
    bars = ax1.barh(y_pos, df['Avg_Placement'], color=bar_colors, 
                    edgecolor='white', height=0.65, linewidth=1.5)
    
    # 条形图内的数值和出场次数标注
    for i, (_, row) in enumerate(df.iterrows()):
        # 出场次数标签
        ax1.text(row['Avg_Placement'] + 0.12, i, f"n = {int(row['Appearances'])}", 
                 va='center', fontsize=10, color='#7F8C8D', fontweight='bold')
        # 精确数值（在条形内部）
        if row['Avg_Placement'] > 1.5:
            ax1.text(row['Avg_Placement'] - 0.15, i, f"{row['Avg_Placement']:.2f}", 
                     va='center', ha='right', fontsize=10, color='white', fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['Partner'], fontsize=12, fontweight='bold', color='#2C3E50')
    ax1.invert_yaxis()  # Best at top
    ax1.set_xlabel('Average Placement (Lower = Better)', fontsize=13, 
                   labelpad=10, color='#34495E')
    ax1.set_xlim(0, 8.5)
    ax1.set_ylim(9.8, -0.6)
    
    # 左轴样式
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#CCCCCC')
    ax1.spines['bottom'].set_color('#CCCCCC')
    
    # Highlight Derek Hough (第一名)
    ax1.get_yticklabels()[0].set_color(COLOR_ACCENT)
    
    # "Champion Effect" 标注框
    first_placement = df['Avg_Placement'].iloc[0]
    ax1.text(first_placement / 2, 0, 'Champion\nEffect', 
             fontsize=11, fontweight='bold', color=COLOR_ACCENT,
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDF2F2', 
                       edgecolor=COLOR_ACCENT, linewidth=1.5))
      # 右轴：Avg_Judge 折线图
    ax2 = ax1.twiny()  # 共享y轴，创建顶部x轴
    ax2.set_xlim(20, 30.6)  # 根据 Avg_Judge 数据范围调整
    
    # 绘制 Avg_Judge 折线（橙色 #FA8600）
    COLOR_JUDGE_LINE = '#FA8600'
    ax2.plot(df['Avg_Judge'], y_pos, color=COLOR_JUDGE_LINE, linewidth=2.5, 
             marker='o', markersize=8, markerfacecolor='white', 
             markeredgecolor=COLOR_JUDGE_LINE, markeredgewidth=2, zorder=10)
    
    # 在每个点旁边显示数值
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.text(row['Avg_Judge'] + 0.4, i, f"{row['Avg_Judge']:.1f}", 
                 va='center', ha='left', fontsize=9, color=COLOR_JUDGE_LINE, fontweight='bold')
    
    # 右轴（顶部x轴）样式
    ax2.set_xlabel('Average Judge Score', fontsize=13, labelpad=10, color=COLOR_JUDGE_LINE)
    ax2.xaxis.label.set_color(COLOR_JUDGE_LINE)
    ax2.tick_params(axis='x', colors=COLOR_JUDGE_LINE)
    ax2.spines['top'].set_color(COLOR_JUDGE_LINE)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 标题
    ax1.set_title('Top 10 Professional Dancers: Placement & Judge Scores', 
                  fontsize=16, fontweight='bold', pad=45, color='#2C3E50')
    
    # 添加图例
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#375093', edgecolor='white', label='Avg Placement'),
        Line2D([0], [0], color=COLOR_JUDGE_LINE, linewidth=2.5, marker='o', 
               markersize=8, markerfacecolor='white', markeredgecolor=COLOR_JUDGE_LINE,
               label='Avg Judge Score')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=11, 
               frameon=True, facecolor='white', edgecolor='#CCCCCC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'task3_2_partner_ranking_top10_dual.png'), bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("[VIZ] Saved: task3_2_partner_ranking_top10_dual.png")


# ==== 图4: Judge vs Fan 敏感度对比 ====
def plot_judge_fan_comparison():
    """Compare judge vs fan sensitivity with elegant styling"""
    df = pd.read_csv('../data/task3_regression_coefficients.csv')
    
    judge = df[df['Model'] == 'Judge_Score'].set_index('Variable')
    fan = df[df['Model'] == 'Fan_Vote'].set_index('Variable')
    
    # Select key variables
    key_vars = ['age', 'celeb_popularity', 'partner_popularity', 'dance_experience_score']
    
    judge_t = judge.loc[key_vars, 't_value'].abs()
    fan_t = fan.loc[key_vars, 't_value'].abs()
    
    # Create grouped bar chart
    var_labels = ['Age', 'Celebrity\nPopularity', 'Partner\nPopularity', 'Dance\nExperience']
    x = np.arange(len(var_labels))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('white')
    
    # 使用自定义颜色
    bars1 = ax.bar(x - width/2, judge_t.values, width, label='Judge Score', 
                   color=COLOR_JUDGE, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, fan_t.values, width, label='Fan Vote', 
                   color=COLOR_FAN, edgecolor='white', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(var_labels, fontsize=12, color='#2C3E50')
    ax.set_ylabel('|t-statistic|', fontsize=14, labelpad=10, color='#34495E')
    ax.set_title('Sensitivity Comparison: Judges vs. Fans', 
                 fontsize=16, fontweight='bold', pad=15, color='#2C3E50')
    
    # 图例
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              facecolor='white', edgecolor='#CCCCCC', framealpha=0.9)
    
    # 显著性阈值线
    ax.axhline(y=1.96, color='#95A5A6', linestyle='--', alpha=0.8, linewidth=1.5)
    ax.text(3.55, 2.15, 'p = 0.05', color='#7F8C8D', fontsize=11, style='italic')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=11, fontweight='medium', color=COLOR_JUDGE)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=11, fontweight='medium', color=COLOR_FAN)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    ax.set_ylim(0, max(judge_t.max(), fan_t.max()) * 1.25)
    
    plt.tight_layout()
    plt.savefig('../figures/task3_2_judge_fan_sensitivity.png', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("[VIZ] Saved: task3_2_judge_fan_sensitivity.png")


# ==== 图5: 模型拟合度对比 ====
def plot_model_fit():
    """Plot R-squared comparison with elegant styling"""
    model_stats = pd.read_csv('../data/task3_model_stats.csv')
    
    # Filter to contestant-level models
    contestant_models = model_stats[model_stats['Model'].isin(['Judge_Score', 'Fan_Vote', 'Placement'])]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('white')
    
    models = ['Judge Score', 'Fan Vote', 'Placement']
    r2_values = contestant_models['R2'].values
    adj_r2_values = contestant_models['Adj_R2'].values
    
    x = np.arange(len(models))
    width = 0.36
    
    # 使用蓝色渐变
    bars1 = ax.bar(x - width/2, r2_values, width, label='R²', 
                   color=COLOR_SECONDARY, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, adj_r2_values, width, label='Adjusted R²', 
                   color=COLOR_PRIMARY, edgecolor='white', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, color='#2C3E50')
    ax.set_ylabel('R²', fontsize=13, labelpad=10, color='#34495E')
    ax.set_title('Model Fit: Contestant-Level Regressions', 
                 fontsize=15, fontweight='bold', pad=15, color='#2C3E50')
    ax.legend(loc='upper right', fontsize=11, frameon=True, 
              facecolor='white', edgecolor='#CCCCCC')
    ax.set_ylim(0, 0.35)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=11, fontweight='medium', color=COLOR_SECONDARY)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=11, fontweight='medium', color=COLOR_PRIMARY)
    
    # 参考线：社会科学研究的典型 R² 范围
    ax.axhline(y=0.20, color='#95A5A6', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(2.6, 0.21, 'Typical for\nSocial Science', color='#7F8C8D', 
            fontsize=9, style='italic', ha='center')
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    plt.savefig('../figures/task3_2_model_fit.png', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("[VIZ] Saved: task3_2_model_fit.png")


# ==== 图6: 行业效应对比 ====
def plot_industry_effect():
    """Plot industry effect on different outcomes"""
    df = pd.read_csv('../data/task3_regression_coefficients.csv')
    
    # 筛选行业变量
    industry_vars = ['Industry_[T.Actor]', 'Industry_[T.Athlete]', 'Industry_[T.Singer]',
                     'Industry_[T.SocialMedia]', 'Industry_[T.Model]', 'Industry_[T.Comedian]']
    
    df_ind = df[df['Variable'].isin(industry_vars)]
    df_ind = df_ind[df_ind['Model'].isin(['Judge_Score', 'Fan_Vote', 'Placement'])]
    
    # 简化行业名称
    industry_names = {
        'Industry_[T.Actor]': 'Actor',
        'Industry_[T.Athlete]': 'Athlete', 
        'Industry_[T.Singer]': 'Singer',
        'Industry_[T.SocialMedia]': 'Social Media',
        'Industry_[T.Model]': 'Model',
        'Industry_[T.Comedian]': 'Comedian'
    }
    df_ind['Industry'] = df_ind['Variable'].map(industry_names)
    
    # Pivot
    coef_pivot = df_ind.pivot(index='Industry', columns='Model', values='Coefficient')
    pval_pivot = df_ind.pivot(index='Industry', columns='Model', values='p_value')
    
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(coef_pivot.index))
    width = 0.26
    
    # 三个模型的系数
    colors = [COLOR_JUDGE, COLOR_FAN, COLOR_PLACEMENT]
    model_labels = ['Judge Score', 'Fan Vote', 'Placement']
    
    for i, (model, color, label) in enumerate(zip(['Judge_Score', 'Fan_Vote', 'Placement'], colors, model_labels)):
        if model in coef_pivot.columns:
            vals = coef_pivot[model].values
            bars = ax.bar(x + (i-1)*width, vals, width, label=label, 
                         color=color, edgecolor='white', linewidth=1.2)
            
            # 添加显著性标记
            for j, (val, idx) in enumerate(zip(vals, coef_pivot.index)):
                pval = pval_pivot.loc[idx, model]
                if pval < 0.01:
                    marker = '***'
                elif pval < 0.05:
                    marker = '**'
                elif pval < 0.10:
                    marker = '*'
                else:
                    marker = ''
                if marker:
                    y_pos = val + 0.15 if val > 0 else val - 0.35
                    ax.text(x[j] + (i-1)*width, y_pos, marker, ha='center', 
                            fontsize=11, color=color, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(coef_pivot.index, fontsize=12, color='#2C3E50')
    ax.set_ylabel('Coefficient (vs. Baseline)', fontsize=14, labelpad=10, color='#34495E')
    ax.set_title('Industry Effect on Competition Outcomes', 
                 fontsize=16, fontweight='bold', pad=15, color='#2C3E50')
    
    ax.axhline(y=0, color='#2C3E50', linewidth=1, alpha=0.5)
    ax.legend(loc='upper left', fontsize=11, frameon=True, 
              facecolor='white', edgecolor='#CCCCCC')
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    plt.savefig('../figures/task3_2_industry_effect.png', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("[VIZ] Saved: task3_2_industry_effect.png")


# ==== 主函数 ====
if __name__ == '__main__':
    print("=" * 60)
    print("Task 3 Final Visualizations (Times New Roman)")
    print("=" * 60)
    # plot_coefficient_heatmap()
    # plot_variable_correlation()
    # plot_partner_top10()
    plot_partner_top10_2()  # 双轴图：条形图 + Avg_Judge 折线
    # plot_judge_fan_comparison()
    # plot_industry_effect()
    
    print("\n" + "=" * 60)
    print("[DONE] All visualizations generated!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - figures/task3_2_coefficient_heatmap.png")
    print("  - figures/task3_2_variable_correlation.png")
    print("  - figures/task3_2_partner_ranking_top10.png")
    print("  - figures/task3_2_partner_ranking_top10_dual.png")
    print("  - figures/task3_2_judge_fan_sensitivity.png")
    print("  - figures/task3_2_industry_effect.png")
