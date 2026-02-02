"""
Task 3: Mixed-Effects Model 完整参数提取与可视化
================================================
提取每个舞伴的随机效应 u_j 并可视化

Author: Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import os
import warnings
warnings.filterwarnings('ignore')

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# 配色
COLOR_PRIMARY = '#375093'
COLOR_SECONDARY = '#7091C7'
COLOR_ACCENT = '#831A21'
COLOR_POSITIVE = '#2E7D32'
COLOR_NEGATIVE = '#C62828'


def load_data():
    """加载选手汇总数据"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'task3_contestant_summary_v3.csv'))
    # 统一列名
    df = df.rename(columns={
        'ballroom_partner': 'Partner',
        'avg_fan_vote_share': 'avg_fan_vote',
        'industry_group': 'Industry'
    })
    print(f"[DATA] Loaded {len(df)} contestants")
    print(f"[DATA] Unique partners: {df['Partner'].nunique()}")
    return df


def fit_mixed_effects_models(df):
    """拟合三个 Mixed-Effects Models 并提取参数"""
    
    results = {}
    random_effects_all = {}
    
    # 准备数据
    df_clean = df.dropna(subset=['avg_judge_score', 'avg_fan_vote', 'placement', 
                                  'Partner', 'age', 'bmi', 'dance_experience_score', 'is_us'])
    
    # 添加行业哑变量
    df_clean = pd.get_dummies(df_clean, columns=['Industry'], drop_first=True, dtype=float)
    
    # 固定效应变量
    fixed_vars = ['age', 'bmi', 'dance_experience_score', 'is_us']
    industry_cols = [col for col in df_clean.columns if col.startswith('Industry_')]
    fixed_vars.extend(industry_cols)
    
    X = df_clean[fixed_vars].astype(float)
    X = sm.add_constant(X)
    
    # 三个因变量
    outcomes = {
        'Judge_Score': 'avg_judge_score',
        'Fan_Vote': 'avg_fan_vote', 
        'Placement': 'placement'
    }
    
    for model_name, y_col in outcomes.items():
        print(f"\n{'='*50}")
        print(f"Fitting Mixed-Effects Model: {model_name}")
        print(f"{'='*50}")
        
        y = df_clean[y_col].astype(float)
        groups = df_clean['Partner']
        
        # 拟合模型
        model = MixedLM(y, X, groups=groups)
        result = model.fit(method='powell', maxiter=500)
        
        # 提取方差分量
        var_u = result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else float(result.cov_re)
        var_e = result.scale
        icc = var_u / (var_u + var_e)
        
        print(f"\n[VARIANCE COMPONENTS]")
        print(f"  σ²_u (Partner variance): {var_u:.6f}")
        print(f"  σ²_ε (Residual variance): {var_e:.6f}")
        print(f"  ICC: {icc:.4f} ({icc*100:.2f}%)")
        
        # 提取随机效应 (BLUPs)
        random_effects = result.random_effects
        re_df = pd.DataFrame([
            {'Partner': partner, 'u_j': effects.iloc[0] if hasattr(effects, 'iloc') else float(effects)}
            for partner, effects in random_effects.items()
        ])
        re_df = re_df.sort_values('u_j', ascending=True if model_name == 'Placement' else False)
        
        print(f"\n[RANDOM EFFECTS - Top 5 Partners]")
        for i, row in re_df.head(5).iterrows():
            print(f"  {row['Partner']}: u_j = {row['u_j']:+.4f}")
        
        results[model_name] = {
            'var_u': var_u,
            'var_e': var_e,
            'icc': icc,
            'model': result
        }
        random_effects_all[model_name] = re_df
    
    return results, random_effects_all


def plot_variance_decomposition(results):
    """可视化方差分解（饼图）"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    model_names = ['Judge_Score', 'Fan_Vote', 'Placement']
    titles = ['Judge Score', 'Fan Vote Share', 'Placement']
    colors = [(COLOR_PRIMARY, '#C8D6E7'), (COLOR_ACCENT, '#ECD0B4'), (COLOR_SECONDARY, '#E8EDF1')]
    
    for idx, (model_name, title) in enumerate(zip(model_names, titles)):
        ax = axes[idx]
        res = results[model_name]
        
        partner_pct = res['icc'] * 100
        residual_pct = 100 - partner_pct
        
        wedges, texts, autotexts = ax.pie(
            [partner_pct, residual_pct],
            autopct=lambda pct: f'{pct:.1f}%' if pct > 10 else '',
            colors=colors[idx],
            explode=(0.03, 0),
            startangle=90,
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'color': '#2C3E50'}
        )
        
        ax.text(0, 0, f'{partner_pct:.1f}%', ha='center', va='center',
                fontsize=16, fontweight='bold', color=colors[idx][0])
        ax.text(0, -1.35, 'Partner Effect', ha='center', va='center',
                fontsize=10, color=colors[idx][0], fontweight='medium')
        ax.set_title(title, fontweight='bold', fontsize=13, pad=12, color='#2C3E50')
    
    fig.suptitle('Variance Decomposition: Partner vs. Residual', 
                 fontsize=15, fontweight='bold', y=1.02, color='#2C3E50')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_PRIMARY, edgecolor='white', label='Partner Effect ($\\sigma^2_u$)'),
        Patch(facecolor='#E8EDF1', edgecolor='white', label='Residual ($\\sigma^2_\\epsilon$)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'task3_me_variance_decomposition.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n[VIZ] Saved: {out_path}")


def plot_random_effects_distribution(random_effects_all):
    """可视化随机效应分布（直方图）"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('white')
    
    model_names = ['Judge_Score', 'Fan_Vote', 'Placement']
    titles = ['Judge Score', 'Fan Vote Share', 'Placement']
    colors = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_SECONDARY]
    
    for idx, (model_name, title, color) in enumerate(zip(model_names, titles, colors)):
        ax = axes[idx]
        re_df = random_effects_all[model_name]
        
        ax.hist(re_df['u_j'], bins=15, color=color, edgecolor='white', 
                linewidth=1.2, alpha=0.85)
        ax.axvline(x=0, color='#2C3E50', linestyle='--', linewidth=1.5, alpha=0.7)
        
        mean_u = re_df['u_j'].mean()
        std_u = re_df['u_j'].std()
        ax.axvline(x=mean_u, color=color, linestyle='-', linewidth=2, alpha=0.9)
        
        ax.set_xlabel('Random Effect $u_j$', fontsize=11, color='#34495E')
        ax.set_ylabel('Count', fontsize=11, color='#34495E')
        ax.set_title(f'{title}\n(Mean={mean_u:.3f}, SD={std_u:.3f})', 
                     fontsize=12, fontweight='bold', color='#2C3E50')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Distribution of Partner Random Effects ($u_j$)', 
                 fontsize=14, fontweight='bold', y=1.02, color='#2C3E50')
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'task3_me_random_effects_dist.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VIZ] Saved: {out_path}")


def plot_top_bottom_partners(random_effects_all):
    """可视化 Top 10 和 Bottom 10 舞伴的随机效应"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    
    model_names = ['Judge_Score', 'Fan_Vote', 'Placement']
    titles = ['Judge Score', 'Fan Vote Share', 'Placement']
    
    for idx, (model_name, title) in enumerate(zip(model_names, titles)):
        ax = axes[idx]
        re_df = random_effects_all[model_name].copy()
        
        # 对于 Placement，负值更好（排名更高）
        if model_name == 'Placement':
            re_df = re_df.sort_values('u_j', ascending=True)
            top_label = 'Best (Lower Placement)'
            bottom_label = 'Worst (Higher Placement)'
        else:
            re_df = re_df.sort_values('u_j', ascending=False)
            top_label = 'Best (Higher Score)'
            bottom_label = 'Worst (Lower Score)'
        
        # Top 10 和 Bottom 10
        top10 = re_df.head(10)
        bottom10 = re_df.tail(10)
        
        # 合并并绘制
        combined = pd.concat([top10, bottom10])
        y_pos = np.arange(len(combined))
        
        colors = [COLOR_POSITIVE if u < 0 else COLOR_NEGATIVE for u in combined['u_j']] \
                 if model_name == 'Placement' else \
                 [COLOR_POSITIVE if u > 0 else COLOR_NEGATIVE for u in combined['u_j']]
        
        bars = ax.barh(y_pos, combined['u_j'], color=colors, 
                       edgecolor='white', height=0.7, linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(combined['Partner'], fontsize=9, color='#2C3E50')
        ax.invert_yaxis()
        ax.axvline(x=0, color='#2C3E50', linewidth=1.2)
        ax.set_xlabel('Random Effect $u_j$', fontsize=11, color='#34495E')
        ax.set_title(f'{title}', fontsize=13, fontweight='bold', color='#2C3E50', pad=10)
        
        # 添加分隔线
        ax.axhline(y=9.5, color='#CCCCCC', linestyle='--', linewidth=1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Top 10 and Bottom 10 Partners by Random Effect ($u_j$)', 
                 fontsize=15, fontweight='bold', y=1.01, color='#2C3E50')
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'task3_me_top_bottom_partners.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VIZ] Saved: {out_path}")


def plot_placement_random_effects_only(random_effects_all):
    """单独可视化 Placement 模型的 Top 15 舞伴随机效应（论文用）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    re_df = random_effects_all['Placement'].copy()
    re_df = re_df.sort_values('u_j', ascending=True).head(15)  # Top 15 (最低 u_j = 最好)
    
    y_pos = np.arange(len(re_df))
    
    # 渐变色
    n = len(re_df)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n))[::-1]
    
    bars = ax.barh(y_pos, -re_df['u_j'], color=colors,  # 取负值使正向更直观
                   edgecolor='white', height=0.72, linewidth=1.5)
    
    # 添加数值标签
    for i, (_, row) in enumerate(re_df.iterrows()):
        ax.text(-row['u_j'] + 0.05, i, f"{-row['u_j']:.2f}", 
                va='center', fontsize=10, color='#2C3E50', fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(re_df['Partner'], fontsize=11, fontweight='bold', color='#2C3E50')
    ax.invert_yaxis()
    ax.set_xlabel('Partner Advantage (−$u_j$, higher = better)', fontsize=12, 
                  labelpad=10, color='#34495E')
    ax.set_title('Top 15 Professional Dancers by Random Effect on Placement', 
                 fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
    
    # 突出 Derek Hough
    ax.get_yticklabels()[0].set_color(COLOR_ACCENT)
    
    ax.set_xlim(0, max(-re_df['u_j']) * 1.15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'task3_me_placement_top15.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VIZ] Saved: {out_path}")


def save_results_to_csv(results, random_effects_all):
    """保存结果到 CSV"""
    # 方差分量
    var_df = pd.DataFrame([
        {
            'Model': model_name,
            'Var_Partner': res['var_u'],
            'Var_Residual': res['var_e'],
            'ICC': res['icc']
        }
        for model_name, res in results.items()
    ])
    var_path = os.path.join(DATA_DIR, 'task3_me_variance_components.csv')
    var_df.to_csv(var_path, index=False)
    print(f"\n[DATA] Saved: {var_path}")
    
    # 随机效应（所有舞伴）
    all_re = []
    for model_name, re_df in random_effects_all.items():
        re_df = re_df.copy()
        re_df['Model'] = model_name
        all_re.append(re_df)
    
    re_combined = pd.concat(all_re, ignore_index=True)
    re_path = os.path.join(DATA_DIR, 'task3_me_random_effects.csv')
    re_combined.to_csv(re_path, index=False)
    print(f"[DATA] Saved: {re_path}")


def main():
    print("=" * 60)
    print("Task 3: Mixed-Effects Model Parameter Extraction")
    print("=" * 60)
    
    # 加载数据
    df = load_data()
    
    # 拟合模型
    results, random_effects_all = fit_mixed_effects_models(df)
    
    # 生成可视化
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    plot_variance_decomposition(results)
    plot_random_effects_distribution(random_effects_all)
    plot_top_bottom_partners(random_effects_all)
    plot_placement_random_effects_only(random_effects_all)
    
    # 保存数据
    save_results_to_csv(results, random_effects_all)
    
    print("\n" + "=" * 60)
    print("[DONE] All Mixed-Effects Model outputs generated!")
    print("=" * 60)
    
    # 打印汇总
    print("\n[SUMMARY] Generated files:")
    print("  - figures/task3_me_variance_decomposition.png")
    print("  - figures/task3_me_random_effects_dist.png")
    print("  - figures/task3_me_top_bottom_partners.png")
    print("  - figures/task3_me_placement_top15.png")
    print("  - data/task3_me_variance_components.csv")
    print("  - data/task3_me_random_effects.csv")


if __name__ == '__main__':
    main()
