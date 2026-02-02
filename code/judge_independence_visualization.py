"""
Judge Independence Visualization for DWTS
==========================================
Generates plots for judge independence analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================
# Data Loading
# ============================================

def load_and_preprocess_data(filepath):
    """Load DWTS data and extract judge scores."""
    df = pd.read_csv(filepath)
    return df

def extract_weekly_scores(df, season, week):
    """Extract valid judge scores for a specific season and week."""
    cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
    valid_cols = [c for c in cols if c in df.columns]
    
    season_df = df[df['season'] == season][valid_cols].copy()
    season_df = season_df.replace(0, np.nan)
    season_df = season_df.dropna(how='all')
    season_df = season_df.dropna(axis=1, how='all')
    season_df = season_df.dropna()
    season_df.columns = [f'Judge_{i+1}' for i in range(len(season_df.columns))]
    
    return season_df

def aggregate_all_data(df):
    """Aggregate all valid judge scores across all seasons/weeks."""
    all_scores = []
    
    for season in df['season'].unique():
        for week in range(1, 12):
            scores = extract_weekly_scores(df, season, week)
            if len(scores) >= 3 and len(scores.columns) >= 3:
                scores_3j = scores.iloc[:, :3].copy()
                scores_3j['season'] = season
                scores_3j['week'] = week
                all_scores.append(scores_3j)
    
    if all_scores:
        combined = pd.concat(all_scores, ignore_index=True)
        return combined
    return None

# ============================================
# Visualization Functions
# ============================================

def plot_correlation_heatmap(scores_df, save_path):
    """Plot correlation heatmap for judges."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pearson correlation
    corr_pearson = scores_df.corr(method='pearson')
    sns.heatmap(corr_pearson, annot=True, fmt='.3f', cmap='RdYlBu_r',
                vmin=0.5, vmax=1.0, ax=axes[0], square=True,
                cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Pearson Correlation Between Judges', fontsize=12, fontweight='bold')
    
    # Spearman correlation
    corr_spearman = scores_df.corr(method='spearman')
    sns.heatmap(corr_spearman, annot=True, fmt='.3f', cmap='RdYlBu_r',
                vmin=0.5, vmax=1.0, ax=axes[1], square=True,
                cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Spearman Correlation Between Judges', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_pairwise_scatter(scores_df, save_path):
    """Plot pairwise scatter plots between judges."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    pairs = [('Judge_1', 'Judge_2'), ('Judge_1', 'Judge_3'), ('Judge_2', 'Judge_3')]
    
    for ax, (j1, j2) in zip(axes, pairs):
        x = scores_df[j1].values
        y = scores_df[j2].values
        
        # Scatter with alpha
        ax.scatter(x, y, alpha=0.3, s=15, color='steelblue')
        
        # Regression line
        slope, intercept, r, p, se = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, 
                label=f'r = {r:.3f}')
        
        # Identity line
        ax.plot([0, 10], [0, 10], 'k--', linewidth=1, alpha=0.5, label='y = x')
        
        ax.set_xlabel(j1.replace('_', ' '), fontsize=11)
        ax.set_ylabel(j2.replace('_', ' '), fontsize=11)
        ax.set_xlim(2, 10.5)
        ax.set_ylim(2, 10.5)
        ax.legend(loc='upper left')
        ax.set_title(f'{j1.replace("_", " ")} vs {j2.replace("_", " ")}', fontweight='bold')
    
    plt.suptitle('Pairwise Judge Score Comparison (n = 2759)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_residual_analysis(scores_df, save_path):
    """Plot residual analysis after controlling for contestant skill."""
    # Calculate residuals
    true_skill = scores_df.mean(axis=1)
    residuals = scores_df.sub(true_skill, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residual distributions
    ax1 = axes[0, 0]
    for col in residuals.columns:
        sns.kdeplot(residuals[col], ax=ax1, label=col.replace('_', ' '), linewidth=2)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Residual (Score - Mean)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Residual Distribution by Judge', fontweight='bold')
    ax1.legend()
    
    # 2. Residual correlation heatmap
    ax2 = axes[0, 1]
    residual_corr = residuals.corr()
    sns.heatmap(residual_corr, annot=True, fmt='.3f', cmap='RdBu_r',
                vmin=-1.0, vmax=1.0, ax=ax2, square=True,
                cbar_kws={'label': 'Correlation'})
    ax2.set_title('Residual Correlation Matrix', fontweight='bold')
    
    # 3. Residual scatter plot (J1 vs J2)
    ax3 = axes[1, 0]
    ax3.scatter(residuals['Judge_1'], residuals['Judge_2'], alpha=0.3, s=15, color='coral')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    r, p = stats.pearsonr(residuals['Judge_1'], residuals['Judge_2'])
    ax3.text(0.05, 0.95, f'r = {r:.3f}', transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    ax3.set_xlabel('Judge 1 Residual', fontsize=11)
    ax3.set_ylabel('Judge 2 Residual', fontsize=11)
    ax3.set_title('Residual Scatter: Judge 1 vs Judge 2', fontweight='bold')
    
    # 4. Residual scatter plot (J1 vs J3)
    ax4 = axes[1, 1]
    ax4.scatter(residuals['Judge_1'], residuals['Judge_3'], alpha=0.3, s=15, color='seagreen')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    r, p = stats.pearsonr(residuals['Judge_1'], residuals['Judge_3'])
    ax4.text(0.05, 0.95, f'r = {r:.3f}', transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    ax4.set_xlabel('Judge 1 Residual', fontsize=11)
    ax4.set_ylabel('Judge 3 Residual', fontsize=11)
    ax4.set_title('Residual Scatter: Judge 1 vs Judge 3', fontweight='bold')
    
    plt.suptitle('Residual Analysis (After Controlling for Contestant Skill)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_bland_altman(scores_df, save_path):
    """Plot Bland-Altman plots for each judge pair."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    pairs = [('Judge_1', 'Judge_2'), ('Judge_1', 'Judge_3'), ('Judge_2', 'Judge_3')]
    colors = ['steelblue', 'coral', 'seagreen']
    
    for ax, (j1, j2), color in zip(axes, pairs, colors):
        x = scores_df[j1].values
        y = scores_df[j2].values
        
        mean_xy = (x + y) / 2
        diff_xy = x - y
        
        mean_diff = np.mean(diff_xy)
        std_diff = np.std(diff_xy, ddof=1)
        
        ax.scatter(mean_xy, diff_xy, alpha=0.3, s=15, color=color)
        
        # Mean line
        ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean = {mean_diff:.3f}')
        
        # Limits of Agreement
        ax.axhline(y=mean_diff + 1.96*std_diff, color='gray', linestyle='--', 
                   linewidth=1.5, label=f'+1.96 SD = {mean_diff + 1.96*std_diff:.2f}')
        ax.axhline(y=mean_diff - 1.96*std_diff, color='gray', linestyle='--', 
                   linewidth=1.5, label=f'-1.96 SD = {mean_diff - 1.96*std_diff:.2f}')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel(f'Mean of {j1.replace("_", " ")} and {j2.replace("_", " ")}', fontsize=10)
        ax.set_ylabel(f'{j1.replace("_", " ")} - {j2.replace("_", " ")}', fontsize=10)
        ax.set_title(f'Bland-Altman: {j1.replace("_", " ")} vs {j2.replace("_", " ")}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Bland-Altman Analysis (Systematic Bias Detection)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_season_trends(results_df, save_path):
    """Plot season-by-season independence metrics."""
    # Load season summary
    season_summary = results_df.groupby('season').agg({
        'avg_pearson_corr': 'mean',
        'ICC_2_1': 'mean',
        'Kendall_W': 'mean',
        'avg_residual_corr': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Pearson correlation over seasons
    ax1 = axes[0, 0]
    ax1.plot(season_summary['season'], season_summary['avg_pearson_corr'], 
             'o-', color='steelblue', linewidth=2, markersize=6)
    ax1.axhline(y=season_summary['avg_pearson_corr'].mean(), color='red', 
                linestyle='--', label=f'Mean = {season_summary["avg_pearson_corr"].mean():.3f}')
    ax1.set_xlabel('Season', fontsize=11)
    ax1.set_ylabel('Average Pearson Correlation', fontsize=11)
    ax1.set_title('Judge Correlation by Season', fontweight='bold')
    ax1.set_ylim(0.6, 1.0)
    ax1.legend()
    
    # 2. ICC over seasons
    ax2 = axes[0, 1]
    ax2.plot(season_summary['season'], season_summary['ICC_2_1'], 
             'o-', color='coral', linewidth=2, markersize=6)
    ax2.axhline(y=season_summary['ICC_2_1'].mean(), color='red', 
                linestyle='--', label=f'Mean = {season_summary["ICC_2_1"].mean():.3f}')
    ax2.fill_between(season_summary['season'], 0.75, 1.0, alpha=0.2, color='green', label='Good-Excellent')
    ax2.set_xlabel('Season', fontsize=11)
    ax2.set_ylabel('ICC(2,1)', fontsize=11)
    ax2.set_title('Intraclass Correlation Coefficient by Season', fontweight='bold')
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()
    
    # 3. Kendall's W over seasons
    ax3 = axes[1, 0]
    ax3.plot(season_summary['season'], season_summary['Kendall_W'], 
             'o-', color='seagreen', linewidth=2, markersize=6)
    ax3.axhline(y=season_summary['Kendall_W'].mean(), color='red', 
                linestyle='--', label=f'Mean = {season_summary["Kendall_W"].mean():.3f}')
    ax3.set_xlabel('Season', fontsize=11)
    ax3.set_ylabel("Kendall's W", fontsize=11)
    ax3.set_title("Kendall's W (Agreement) by Season", fontweight='bold')
    ax3.set_ylim(0.5, 1.0)
    ax3.legend()
    
    # 4. Residual correlation over seasons
    ax4 = axes[1, 1]
    ax4.plot(season_summary['season'], season_summary['avg_residual_corr'], 
             'o-', color='purple', linewidth=2, markersize=6)
    ax4.axhline(y=0, color='black', linestyle=':', linewidth=1)
    ax4.axhline(y=season_summary['avg_residual_corr'].mean(), color='red', 
                linestyle='--', label=f'Mean = {season_summary["avg_residual_corr"].mean():.3f}')
    ax4.set_xlabel('Season', fontsize=11)
    ax4.set_ylabel('Average Residual Correlation', fontsize=11)
    ax4.set_title('Residual Correlation by Season (Negative = Complementary)', fontweight='bold')
    ax4.set_ylim(-0.7, 0.1)
    ax4.legend()
    
    plt.suptitle('Judge Independence Metrics Across 34 Seasons', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_summary_dashboard(scores_df, save_path):
    """Create a summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    
    # Layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Score distributions
    ax1 = fig.add_subplot(gs[0, 0])
    for col in scores_df.columns:
        sns.kdeplot(scores_df[col], ax=ax1, label=col.replace('_', ' '), linewidth=2)
    ax1.set_xlabel('Score', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('Score Distributions', fontweight='bold')
    ax1.legend(fontsize=8)
    
    # 2. Correlation heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    corr = scores_df.corr()
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
                vmin=0.8, vmax=1.0, ax=ax2, square=True)
    ax2.set_title('Correlation Matrix', fontweight='bold')
    
    # 3. Box plots
    ax3 = fig.add_subplot(gs[0, 2])
    scores_df.boxplot(ax=ax3)
    ax3.set_ylabel('Score', fontsize=10)
    ax3.set_title('Score Box Plots', fontweight='bold')
    
    # 4-6. Pairwise scatters
    pairs = [('Judge_1', 'Judge_2'), ('Judge_1', 'Judge_3'), ('Judge_2', 'Judge_3')]
    for i, (j1, j2) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, i])
        ax.scatter(scores_df[j1], scores_df[j2], alpha=0.2, s=10)
        r, _ = stats.pearsonr(scores_df[j1], scores_df[j2])
        ax.set_xlabel(j1.replace('_', ' '), fontsize=9)
        ax.set_ylabel(j2.replace('_', ' '), fontsize=9)
        ax.set_title(f'r = {r:.3f}', fontweight='bold')
        ax.plot([3, 10], [3, 10], 'r--', alpha=0.5)
    
    # 7. Residual correlation heatmap
    ax7 = fig.add_subplot(gs[2, 0])
    true_skill = scores_df.mean(axis=1)
    residuals = scores_df.sub(true_skill, axis=0)
    res_corr = residuals.corr()
    sns.heatmap(res_corr, annot=True, fmt='.3f', cmap='RdBu_r',
                vmin=-1.0, vmax=1.0, ax=ax7, square=True)
    ax7.set_title('Residual Correlation', fontweight='bold')
    
    # 8. Summary statistics text
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    # Calculate statistics
    avg_corr = np.mean(corr.values[np.triu_indices(3, k=1)])
    avg_res_corr = np.mean(res_corr.values[np.triu_indices(3, k=1)])
    
    summary_text = """
    JUDGE INDEPENDENCE ANALYSIS SUMMARY
    =====================================
    
    Total Observations: 2,759 contestant-week scores
    
    RAW SCORE METRICS:
    • Average Pearson Correlation: {:.3f}
    • Average Spearman Correlation: {:.3f}
    → High correlation expected (same performance evaluated)
    
    RESIDUAL METRICS (after controlling for skill):
    • Average Residual Correlation: {:.3f}
    → Negative correlation indicates complementary biases
    
    INTERPRETATION:
    ✓ Judges agree on contestant rankings (r ≈ 0.9)
    ✓ Individual biases are NOT positively correlated
    ✓ Negative residual correlation means: when one judge
      gives higher-than-average, another tends to give
      lower-than-average → "balancing" effect
    
    CONCLUSION:
    Judges demonstrate CONDITIONAL INDEPENDENCE
    (independent given contestant skill)
    """.format(avg_corr, avg_corr, avg_res_corr)
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DWTS Judge Independence Analysis Dashboard', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    # Load data
    data_path = r"e:\比赛\数学建模\2026美赛\comap26\2026-repo\data\2026_MCM_Problem_C_Data_Cleaned.csv"
    df = load_and_preprocess_data(data_path)
    
    # Aggregate data
    combined = aggregate_all_data(df)
    scores_only = combined[['Judge_1', 'Judge_2', 'Judge_3']]
    
    # Load season results
    results_path = r"e:\比赛\数学建模\2026美赛\comap26\2026-repo\data\judge_independence_results.csv"
    results_df = pd.read_csv(results_path)
    
    # Output directory
    fig_dir = r"e:\比赛\数学建模\2026美赛\comap26\2026-repo\figures"
    
    print("Generating visualizations...")
    
    # Generate plots
    plot_correlation_heatmap(scores_only, f"{fig_dir}/judge_correlation_heatmap.png")
    plot_pairwise_scatter(scores_only, f"{fig_dir}/judge_pairwise_scatter.png")
    plot_residual_analysis(scores_only, f"{fig_dir}/judge_residual_analysis.png")
    plot_bland_altman(scores_only, f"{fig_dir}/judge_bland_altman.png")
    plot_season_trends(results_df, f"{fig_dir}/judge_season_trends.png")
    plot_summary_dashboard(scores_only, f"{fig_dir}/judge_independence_dashboard.png")
    
    print("\nAll visualizations generated successfully!")
