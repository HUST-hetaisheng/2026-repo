"""
Task 3 Visualization V4 - Updated for corrected models
Generates figures with consistent styling for the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, 'data')
FIG_DIR = os.path.join(REPO_DIR, 'figures')

# Style settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

def plot_judge_fan_sensitivity():
    """
    Compare judge vs fan sensitivity to different factors.
    Uses t-statistics from V4 regression results.
    """
    
    # Load coefficient data
    coef_df = pd.read_csv(os.path.join(DATA_DIR, 'task3_regression_coefficients.csv'))
    
    # Extract t-values for contestant-level models
    judge_df = coef_df[coef_df['Model'] == 'Judge_Score'].set_index('Variable')
    fan_df = coef_df[coef_df['Model'] == 'Fan_Vote'].set_index('Variable')
    
    # For Weekly models (to get social_media_popularity)
    weekly_fan_df = coef_df[coef_df['Model'] == 'Weekly_Fan'].set_index('Variable')
    weekly_judge_df = coef_df[coef_df['Model'] == 'Weekly_Judge'].set_index('Variable')
    
    # Variables to compare (using Weekly model for social_media)
    variables = {
        'Age': ('age', 'contestant'),
        'Social Media Popularity': ('social_media_popularity', 'weekly'),
        'Dance Experience': ('dance_experience_score', 'contestant'),
        'BMI': ('bmi', 'contestant'),
        'is_US': ('is_us', 'contestant'),
    }
    
    judge_t = []
    fan_t = []
    labels = []
    
    for label, (var, model_type) in variables.items():
        if model_type == 'contestant':
            if var in judge_df.index and var in fan_df.index:
                judge_t.append(judge_df.loc[var, 't_value'])
                fan_t.append(fan_df.loc[var, 't_value'])
                labels.append(label)
        else:  # weekly
            if var in weekly_judge_df.index and var in weekly_fan_df.index:
                judge_t.append(weekly_judge_df.loc[var, 't_value'])
                fan_t.append(weekly_fan_df.loc[var, 't_value'])
                labels.append(label)
    
    # Convert to arrays
    judge_t = np.array(judge_t)
    fan_t = np.array(fan_t)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Colors
    judge_color = '#3498db'  # Blue
    fan_color = '#e74c3c'    # Red
    
    bars1 = ax.bar(x - width/2, judge_t, width, label='Judge Score', color=judge_color, alpha=0.85)
    bars2 = ax.bar(x + width/2, fan_t, width, label='Fan Vote', color=fan_color, alpha=0.85)
    
    # Add significance lines
    ax.axhline(y=1.96, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-1.96, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(labels)-0.5, 2.2, 'p=0.05', fontsize=10, color='gray')
    
    ax.axhline(y=2.576, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.axhline(y=-2.576, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    
    # Labels
    ax.set_ylabel('t-statistic', fontsize=14)
    ax.set_xlabel('Factor', fontsize=14)
    ax.set_title('Sensitivity Comparison: Judges vs Fans', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(loc='upper right')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.5:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height > 0 else -12),
                            textcoords="offset points",
                            ha='center', va='bottom' if height > 0 else 'top',
                            fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    ax.set_ylim(-15, 35)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(FIG_DIR, 'task3_2_judge_fan_sensitivity.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {output_path}")


def plot_partner_ranking():
    """
    Plot top 10 professional dancers by average placement.
    """
    
    # Load partner ranking data
    partner_df = pd.read_csv(os.path.join(DATA_DIR, 'task3_partner_ranking_v3.csv'))
    partner_df = partner_df.head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(partner_df))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(partner_df)))[::-1]
    
    bars = ax.barh(y_pos, partner_df['Avg_Placement'], color=colors, edgecolor='white', height=0.7)
    
    # Add annotations
    for i, (_, row) in enumerate(partner_df.iterrows()):
        ax.text(row['Avg_Placement'] + 0.1, i, f"n={row['Appearances']:.0f}", 
                va='center', fontsize=10, color='gray')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(partner_df['Partner'])
    ax.invert_yaxis()
    ax.set_xlabel('Average Placement (Lower = Better)', fontsize=12)
    ax.set_title('Top 10 Professional Dancers by Average Placement', fontsize=14, fontweight='bold')
    
    # Highlight Derek Hough
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.get_yticklabels()[0].set_color('#c0392b')
    
    ax.set_xlim(0, 8)
    mean_placement = partner_df['Avg_Placement'].mean()
    ax.axvline(x=mean_placement, color='red', linestyle='--', alpha=0.5, label=f'Mean ({mean_placement:.2f})')
    ax.legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(FIG_DIR, 'task3_2_partner_ranking_top10.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {output_path}")


if __name__ == '__main__':
    print("=" * 50)
    print("Task 3 Visualization V4")
    print("=" * 50)
    
    plot_judge_fan_sensitivity()
    plot_partner_ranking()
    
    print("\n[DONE] All figures generated!")
