"""
Task 3: Visualization for Contestant and Partner Analysis
==========================================================
2026 MCM Problem C - DWTS Analysis

Generates publication-ready figures for Task 3 analysis.

Author: Team
Date: 2026-01
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
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

def load_data():
    """Load analysis data"""
    contestant_df = pd.read_csv('../data/task3_contestant_summary.csv')
    pop_df = pd.read_csv('../data/2026_MCM_Problem_C_Data_Cleaned添加人气后.csv')
    fan_df = pd.read_csv('../data/fan_vote_results_final.csv')
    return contestant_df, pop_df, fan_df


def plot_age_effect(df, save_path='../figures/task3_age_effect.png'):
    """
    Plot age effect on judge scores and fan votes
    Shows scatter with polynomial fit
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    valid_df = df.dropna(subset=['age', 'avg_judge_score', 'avg_fan_vote_share'])
    valid_df = valid_df[valid_df['avg_judge_score'] > 0]
    
    # Age vs Judge Score
    ax1 = axes[0]
    ax1.scatter(valid_df['age'], valid_df['avg_judge_score'], alpha=0.5, s=30, c='steelblue', edgecolor='white')
    
    # Fit polynomial
    z = np.polyfit(valid_df['age'], valid_df['avg_judge_score'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['age'].min(), valid_df['age'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Quadratic fit')
    
    # Mark optimal age
    optimal_age = -z[1] / (2 * z[0])
    if 18 < optimal_age < 75:
        ax1.axvline(optimal_age, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.annotate(f'Optimal: {optimal_age:.0f}', xy=(optimal_age, p(optimal_age)),
                     xytext=(optimal_age + 5, p(optimal_age) + 2),
                     fontsize=10, color='green',
                     arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    ax1.set_xlabel('Age During Season', fontsize=12)
    ax1.set_ylabel('Average Judge Score', fontsize=12)
    ax1.set_title('(A) Age Effect on Judge Scores', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Age vs Fan Vote
    ax2 = axes[1]
    ax2.scatter(valid_df['age'], valid_df['avg_fan_vote_share'], alpha=0.5, s=30, c='coral', edgecolor='white')
    
    z2 = np.polyfit(valid_df['age'], valid_df['avg_fan_vote_share'], 2)
    p2 = np.poly1d(z2)
    ax2.plot(x_line, p2(x_line), 'r-', linewidth=2, label='Quadratic fit')
    
    ax2.set_xlabel('Age During Season', fontsize=12)
    ax2.set_ylabel('Average Fan Vote Share', fontsize=12)
    ax2.set_title('(B) Age Effect on Fan Votes', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VIZ] Saved age effect plot: {save_path}")


def plot_popularity_effect(df, save_path='../figures/task3_popularity_effect.png'):
    """
    Plot celebrity popularity effect on judge scores and fan votes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    valid_df = df.dropna(subset=['celeb_pop_log', 'avg_judge_score', 'avg_fan_vote_share'])
    valid_df = valid_df[valid_df['avg_judge_score'] > 0]
    valid_df = valid_df[valid_df['celeb_pop_log'] > 0]  # Only celebs with popularity data
    
    # Popularity vs Judge Score
    ax1 = axes[0]
    ax1.scatter(valid_df['celeb_pop_log'], valid_df['avg_judge_score'], alpha=0.5, s=30, c='steelblue', edgecolor='white')
    
    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_df['celeb_pop_log'], valid_df['avg_judge_score'])
    x_line = np.linspace(valid_df['celeb_pop_log'].min(), valid_df['celeb_pop_log'].max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label=f'r = {r_value:.3f}')
    
    ax1.set_xlabel('log(Celebrity Popularity + 1)', fontsize=12)
    ax1.set_ylabel('Average Judge Score', fontsize=12)
    ax1.set_title('(A) Popularity vs Judge Scores', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Popularity vs Fan Vote
    ax2 = axes[1]
    ax2.scatter(valid_df['celeb_pop_log'], valid_df['avg_fan_vote_share'], alpha=0.5, s=30, c='coral', edgecolor='white')
    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(valid_df['celeb_pop_log'], valid_df['avg_fan_vote_share'])
    ax2.plot(x_line, slope2 * x_line + intercept2, 'r-', linewidth=2, label=f'r = {r_value2:.3f}')
    
    ax2.set_xlabel('log(Celebrity Popularity + 1)', fontsize=12)
    ax2.set_ylabel('Average Fan Vote Share', fontsize=12)
    ax2.set_title('(B) Popularity vs Fan Votes', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VIZ] Saved popularity effect plot: {save_path}")


def plot_partner_ranking(df, save_path='../figures/task3_partner_ranking.png'):
    """
    Plot top professional dancers by performance
    """
    # Compute partner statistics
    partner_stats = df.groupby('ballroom_partner').agg({
        'placement': ['mean', 'min', 'count'],
        'avg_judge_score': 'mean',
        'avg_fan_vote_share': 'mean'
    }).reset_index()
    partner_stats.columns = ['Partner', 'Avg_Placement', 'Best_Placement', 'Appearances', 'Avg_Judge', 'Avg_Fan']
    partner_stats = partner_stats[partner_stats['Appearances'] >= 3]
    partner_stats = partner_stats.sort_values('Avg_Placement').head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(partner_stats))
    bars = ax.barh(y_pos, partner_stats['Avg_Placement'], color='steelblue', edgecolor='white', height=0.7)
    
    # Add appearance count as text
    for i, (idx, row) in enumerate(partner_stats.iterrows()):
        ax.text(row['Avg_Placement'] + 0.1, i, f"n={row['Appearances']:.0f}", va='center', fontsize=9, color='gray')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(partner_stats['Partner'])
    ax.invert_yaxis()  # Best at top
    ax.set_xlabel('Average Placement (Lower is Better)', fontsize=12)
    ax.set_title('Top 15 Professional Dancers by Average Partner Placement\n(Minimum 3 Appearances)', fontsize=14, fontweight='bold')
    
    # Add vertical line at mean
    ax.axvline(partner_stats['Avg_Placement'].mean(), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Mean')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VIZ] Saved partner ranking plot: {save_path}")


def plot_judge_fan_comparison(df, save_path='../figures/task3_judge_fan_scatter.png'):
    """
    Scatter plot comparing judge scores vs fan votes
    """
    valid_df = df.dropna(subset=['avg_judge_score', 'avg_fan_vote_share'])
    valid_df = valid_df[valid_df['avg_judge_score'] > 0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(valid_df['avg_judge_score'], valid_df['avg_fan_vote_share'], 
                         alpha=0.6, s=40, c=valid_df['placement'], cmap='RdYlGn_r', 
                         edgecolor='white', linewidth=0.5)
    
    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        valid_df['avg_judge_score'], valid_df['avg_fan_vote_share'])
    x_line = np.linspace(valid_df['avg_judge_score'].min(), valid_df['avg_judge_score'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label=f'r = {r_value:.3f}')
    
    ax.set_xlabel('Average Judge Score', fontsize=12)
    ax.set_ylabel('Average Fan Vote Share', fontsize=12)
    ax.set_title(f'Judge Scores vs Fan Votes\n(Pearson r = {r_value:.3f}, p < 0.001)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Add colorbar for placement
    cbar = plt.colorbar(scatter, ax=ax, label='Final Placement')
    cbar.ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VIZ] Saved judge-fan comparison plot: {save_path}")


def plot_coefficient_comparison(save_path='../figures/task3_coefficient_comparison.png'):
    """
    Bar chart comparing standardized coefficients between judge and fan models
    """
    # These values are from the OLS regression output
    # Updated manually based on analysis results
    variables = ['Age', 'Age²', 'Is US', 'Celebrity\nPopularity', 'Partner\nPopularity']
    judge_t = [-3.84, 2.01, 0.16, -1.18, 1.41]  # t-statistics
    fan_t = [-2.20, 0.68, -0.28, 2.85, -2.88]   # t-statistics
    
    x = np.arange(len(variables))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, judge_t, width, label='Judge Score Model', color='steelblue', edgecolor='white')
    bars2 = ax.bar(x + width/2, fan_t, width, label='Fan Vote Model', color='coral', edgecolor='white')
    
    ax.axhline(y=1.96, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=-1.96, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    ax.annotate('p < 0.05', xy=(4.5, 2.1), fontsize=9, color='gray')
    ax.annotate('p < 0.05', xy=(4.5, -2.3), fontsize=9, color='gray')
    
    ax.set_ylabel('t-statistic', fontsize=12)
    ax.set_xlabel('Variable', fontsize=12)
    ax.set_title('Coefficient Comparison: Judge Scores vs Fan Votes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variables)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VIZ] Saved coefficient comparison plot: {save_path}")


def plot_icc_comparison(save_path='../figures/task3_icc_comparison.png'):
    """
    Bar chart comparing ICC values between models
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['Judge Score Model', 'Fan Vote Model']
    icc_values = [13.7, 15.3]  # From mixed effects analysis
    
    bars = ax.bar(models, icc_values, color=['steelblue', 'coral'], edgecolor='white', width=0.5)
    
    for bar, icc in zip(bars, icc_values):
        height = bar.get_height()
        ax.annotate(f'{icc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Intraclass Correlation (ICC) %', fontsize=12)
    ax.set_title('Professional Dancer Effect (ICC)\n% of Variance Explained by Partner Choice', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 25)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VIZ] Saved ICC comparison plot: {save_path}")


def plot_partner_experience_effect(df, save_path='../figures/task3_partner_experience.png'):
    """
    Plot effect of partner experience on outcomes
    """
    # Compute partner experience
    partner_seasons = df.groupby('ballroom_partner')['season'].count().reset_index()
    partner_seasons.columns = ['ballroom_partner', 'total_appearances']
    
    # Merge back
    df_exp = df.merge(partner_seasons, on='ballroom_partner')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group by experience level
    df_exp['exp_group'] = pd.cut(df_exp['total_appearances'], bins=[0, 2, 5, 10, 100], 
                                  labels=['1-2', '3-5', '6-10', '10+'])
    
    # Box plot for judge scores
    ax1 = axes[0]
    df_exp.boxplot(column='avg_judge_score', by='exp_group', ax=ax1)
    ax1.set_xlabel('Partner Career Appearances', fontsize=12)
    ax1.set_ylabel('Average Judge Score', fontsize=12)
    ax1.set_title('(A) Judge Scores by Partner Experience', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    # Box plot for placement
    ax2 = axes[1]
    df_exp.boxplot(column='placement', by='exp_group', ax=ax2)
    ax2.set_xlabel('Partner Career Appearances', fontsize=12)
    ax2.set_ylabel('Final Placement (Lower is Better)', fontsize=12)
    ax2.set_title('(B) Placement by Partner Experience', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VIZ] Saved partner experience plot: {save_path}")


def main():
    """Generate all visualizations"""
    
    print("=" * 60)
    print("Task 3: Generating Visualizations")
    print("=" * 60)
    
    # Load data
    print("\n[LOAD] Loading data...")
    contestant_df, pop_df, fan_df = load_data()
    
    # Generate plots
    print("\n[PLOT] Generating figures...")
    
    plot_age_effect(contestant_df)
    plot_popularity_effect(contestant_df)
    plot_partner_ranking(contestant_df)
    plot_judge_fan_comparison(contestant_df)
    plot_coefficient_comparison()
    plot_icc_comparison()
    plot_partner_experience_effect(contestant_df)
    
    print("\n" + "=" * 60)
    print("[DONE] All visualizations saved to ../figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
