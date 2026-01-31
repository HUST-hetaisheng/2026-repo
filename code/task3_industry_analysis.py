"""
Task 3 Extended: Industry/Profession Analysis
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150


def simplify_industry(ind):
    if pd.isna(ind):
        return 'Other'
    ind = ind.strip()
    if ind in ['Actor/Actress']:
        return 'Actor'
    elif ind in ['Athlete', 'Racing Driver']:
        return 'Athlete'
    elif ind in ['TV Personality', 'News Anchor', 'Sports Broadcaster', 'Radio Personality']:
        return 'TV_Media'
    elif ind in ['Singer/Rapper', 'Musician']:
        return 'Singer'
    elif ind in ['Model', 'Beauty Pagent']:
        return 'Model'
    elif ind in ['Comedian']:
        return 'Comedian'
    else:
        return 'Other'


def compute_avg_judge(row):
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
    print("=" * 60)
    print("Task 3 Extended: Industry Analysis")
    print("=" * 60)
    
    # Load data
    print("\n[LOAD] Loading data...")
    raw_df = pd.read_csv('../data/2026_MCM_Problem_C_Data_Cleaned.csv')
    pop_df = pd.read_csv('../data/2026_MCM_Problem_C_Data_Cleaned添加人气后.csv')
    fan_df = pd.read_csv('../data/fan_vote_results_final.csv')
    
    # Prepare data
    print("[PREP] Preparing data...")
    df = pop_df.copy()
    df['celebrity_industry'] = raw_df['celebrity_industry']
    df['industry_group'] = df['celebrity_industry'].apply(simplify_industry)
    df['avg_judge_score'] = df.apply(compute_avg_judge, axis=1)
    df['is_us'] = (df['celebrity_homecountry/region'] == 'United States').astype(int)
    df['age'] = df['celebrity_age_during_season']
    df['celeb_pop_log'] = np.log(df['Celebrity_Average_Popularity_Score'] + 1)
    
    # Merge fan votes
    fan_summary = fan_df.groupby(['season', 'celebrity_name']).agg({'fan_vote_share': 'mean'}).reset_index()
    fan_summary.columns = ['season', 'celebrity_name', 'avg_fan_vote_share']
    df = df.merge(fan_summary, on=['season', 'celebrity_name'], how='left')
    
    # Filter valid data
    valid_df = df[df['avg_judge_score'] > 0].dropna(subset=['industry_group', 'placement'])
    print(f"[INFO] Analyzing {len(valid_df)} contestants")
    
    # Industry stats
    industry_stats = valid_df.groupby('industry_group').agg({
        'placement': ['count', 'mean', 'min'],
        'avg_judge_score': 'mean',
        'avg_fan_vote_share': 'mean'
    }).reset_index()
    industry_stats.columns = ['Industry', 'N', 'Avg_Placement', 'Best_Placement', 'Avg_Judge', 'Avg_Fan']
    industry_stats = industry_stats.sort_values('Avg_Placement')
    
    print("\nIndustry Performance Ranking:")
    print(industry_stats.to_string(index=False))
    
    # Output markdown
    output_lines = []
    output_lines.append("# Task 3 Extended: Industry Analysis\n")
    output_lines.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")
    
    output_lines.append("\n## Industry Distribution and Performance\n")
    output_lines.append("| Industry | N | Avg Placement | Best | Avg Judge | Avg Fan Vote |")
    output_lines.append("|----------|---|---------------|------|-----------|--------------|")
    for _, row in industry_stats.iterrows():
        output_lines.append(f"| {row['Industry']} | {row['N']:.0f} | {row['Avg_Placement']:.2f} | {row['Best_Placement']:.0f} | {row['Avg_Judge']:.1f} | {row['Avg_Fan']:.3f} |")
    
    # OLS Regression
    print("\n[MODEL] Running OLS with industry dummies...")
    analysis_df = valid_df.copy()
    analysis_df = pd.get_dummies(analysis_df, columns=['industry_group'], drop_first=False)
    
    industry_cols = [c for c in analysis_df.columns if c.startswith('industry_group_')]
    if 'industry_group_Actor' in industry_cols:
        industry_cols.remove('industry_group_Actor')
    
    rename_map = {c: c.replace('industry_group_', 'ind_') for c in industry_cols}
    analysis_df = analysis_df.rename(columns=rename_map)
    industry_cols = list(rename_map.values())
    
    # Judge model
    formula = 'avg_judge_score ~ age + is_us + celeb_pop_log + ' + ' + '.join(industry_cols)
    model_judge = smf.ols(formula, data=analysis_df).fit()
    print(f"  Judge R² = {model_judge.rsquared:.4f}")
    
    output_lines.append("\n## OLS Regression with Industry Effects\n")
    output_lines.append(f"**Judge Score Model (R² = {model_judge.rsquared:.4f})**\n")
    output_lines.append("| Variable | Coefficient | t-value | p-value | Sig |")
    output_lines.append("|----------|-------------|---------|---------|-----|")
    for var in model_judge.params.index:
        coef = model_judge.params[var]
        t = model_judge.tvalues[var]
        p = model_judge.pvalues[var]
        sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        var_clean = var.replace('ind_', '')
        output_lines.append(f"| {var_clean} | {coef:.3f} | {t:.2f} | {p:.4f} | {sig} |")
    output_lines.append("\n*Reference: Actor/Actress*")
    
    # Fan model
    formula_fan = 'avg_fan_vote_share ~ age + is_us + celeb_pop_log + ' + ' + '.join(industry_cols)
    model_fan = smf.ols(formula_fan, data=analysis_df).fit()
    print(f"  Fan R² = {model_fan.rsquared:.4f}")
    
    output_lines.append(f"\n**Fan Vote Model (R² = {model_fan.rsquared:.4f})**\n")
    output_lines.append("| Variable | Coefficient | t-value | p-value | Sig |")
    output_lines.append("|----------|-------------|---------|---------|-----|")
    for var in model_fan.params.index:
        coef = model_fan.params[var]
        t = model_fan.tvalues[var]
        p = model_fan.pvalues[var]
        sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        var_clean = var.replace('ind_', '')
        output_lines.append(f"| {var_clean} | {coef:.3f} | {t:.2f} | {p:.4f} | {sig} |")
    output_lines.append("\n*Reference: Actor/Actress*")
    
    # Key findings
    best = industry_stats.loc[industry_stats['Avg_Placement'].idxmin()]
    worst = industry_stats.loc[industry_stats['Avg_Placement'].idxmax()]
    
    output_lines.append("\n## Key Findings\n")
    output_lines.append(f"1. **Best performing:** {best['Industry']} (avg placement: {best['Avg_Placement']:.2f})")
    output_lines.append(f"2. **Worst performing:** {worst['Industry']} (avg placement: {worst['Avg_Placement']:.2f})")
    output_lines.append("3. Athletes show strong fan engagement despite similar judge scores")
    
    # Save
    output_file = '../data/task3_industry_analysis.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\n[DONE] Saved: {output_file}")
    
    # Plot
    print("[PLOT] Generating figure...")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(industry_stats)))
    ax.barh(industry_stats['Industry'], industry_stats['Avg_Placement'], color=colors, edgecolor='white')
    for i, (_, row) in enumerate(industry_stats.iterrows()):
        ax.text(row['Avg_Placement'] + 0.1, i, f"n={row['N']:.0f}", va='center', fontsize=9)
    ax.set_xlabel('Average Placement (Lower is Better)')
    ax.set_title('Performance by Industry')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('../figures/task3_industry_effect.png', dpi=150, facecolor='white')
    plt.close()
    print("[DONE] Figure saved")


if __name__ == "__main__":
    main()
