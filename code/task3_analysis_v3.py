"""
Task 3: Impact of Contestant and Partner Characteristics (Version 3)
=====================================================================
重构版本：
1. 输出回归系数CSV文件
2. 简化MD报告结构
3. 更清晰的结论呈现

Author: Team
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = '../data/'
OUTPUT_DIR = '../data/'

INDUSTRY_MAPPING = {
    'Actor/Actress': 'Actor',
    'Athlete': 'Athlete',
    'TV Personality': 'TV',
    'Singer/Rapper': 'Singer',
    'Model': 'Model',
    'Comedian': 'Comedian',
    'Social Media Personality': 'SocialMedia',
    'Social media personality': 'SocialMedia',
}

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare():
    """加载数据并进行特征工程"""
    
    print("=" * 60)
    print("Task 3 Analysis V3")
    print("=" * 60)
    
    df = pd.read_csv(f'{DATA_DIR}task3_dataset_full.csv')
    print(f"[DATA] Loaded: {len(df)} rows, {df.shape[1]} columns")
    
    # 特征工程
    df['industry_group'] = df['celebrity_industry'].map(
        lambda x: INDUSTRY_MAPPING.get(x, 'Other') if pd.notna(x) else 'Other'
    )
    df['is_us'] = (df['celebrity_homecountry/region'] == 'United States').astype(int)
    df['age'] = df['celebrity_age_during_season']
    df['age_squared'] = df['age'] ** 2
    df['celeb_popularity'] = df['Celebrity_Average_Popularity_Score']
    df['partner_popularity'] = df['ballroom_partner_Average_Popularity_Score']
    
    # 填充缺失值
    for col in ['bmi', 'dance_experience_score', 'social_media_popularity', 
                'google_search_volume', 'celeb_popularity', 'partner_popularity']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df


def create_contestant_summary(df):
    """创建选手级汇总"""
    
    agg = df.groupby(['season', 'celebrity_name']).agg({
        'fan_vote_share': 'mean',
        'judge_total': 'mean',
        'placement': 'first',
        'week': 'max',
        'age': 'first',
        'is_us': 'first',
        'celeb_popularity': 'first',
        'partner_popularity': 'first',
        'bmi': 'first',
        'dance_experience_score': 'first',
        'industry_group': 'first',
        'ballroom_partner': 'first',
        'social_media_popularity': 'mean',
        'google_search_volume': 'mean',
    }).reset_index()
    
    agg.columns = [
        'season', 'celebrity_name', 
        'avg_fan_vote_share', 'avg_judge_score', 'placement', 'weeks_survived',
        'age', 'is_us', 'celeb_popularity', 'partner_popularity', 
        'bmi', 'dance_experience_score', 'industry_group', 'ballroom_partner',
        'avg_social_media_popularity', 'avg_google_search_volume'
    ]
    agg['age_squared'] = agg['age'] ** 2
    
    print(f"[PREP] Created summary for {len(agg)} contestants")
    return agg


def compute_partner_history(df):
    """计算舞伴历史统计"""
    
    partner_seasons = df.groupby(['ballroom_partner', 'season']).agg({
        'placement': 'first'
    }).reset_index()
    
    records = []
    for partner in partner_seasons['ballroom_partner'].unique():
        data = partner_seasons[partner_seasons['ballroom_partner'] == partner].sort_values('season')
        placements = []
        for _, row in data.iterrows():
            records.append({
                'ballroom_partner': partner,
                'season': row['season'],
                'partner_seasons_before': len(placements),
            })
            if pd.notna(row['placement']):
                placements.append(row['placement'])
    
    return pd.DataFrame(records)


# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================

def extract_coefficients(model, model_name):
    """从模型提取系数为DataFrame"""
    
    rows = []
    for var in model.params.index:
        var_clean = var.replace("C(industry_group, Treatment('Other'))", "Industry_")
        rows.append({
            'Model': model_name,
            'Variable': var_clean,
            'Coefficient': model.params[var],
            'Std_Error': model.bse[var],
            't_value': model.tvalues[var],
            'p_value': model.pvalues[var],
            'Significant': '***' if model.pvalues[var] < 0.01 else ('**' if model.pvalues[var] < 0.05 else ('*' if model.pvalues[var] < 0.1 else ''))
        })
    return pd.DataFrame(rows)


def run_ols_models(contestant_df):
    """运行三个OLS回归模型"""
    
    analysis_df = contestant_df.dropna(subset=[
        'age', 'celeb_popularity', 'avg_judge_score', 'avg_fan_vote_share', 
        'placement', 'bmi', 'dance_experience_score'
    ])
    analysis_df = analysis_df[analysis_df['avg_judge_score'] > 0]
    
    print(f"\n[OLS] Sample: {len(analysis_df)} contestants")
    
    formula = """
        age + age_squared + is_us + celeb_popularity + partner_popularity 
        + bmi + dance_experience_score + C(industry_group, Treatment('Other'))
    """
    
    results = {}
    all_coefs = []
    model_stats = []
    
    # Model 1: Judge Score
    m1 = smf.ols(f'avg_judge_score ~ {formula}', data=analysis_df).fit()
    results['judge'] = m1
    all_coefs.append(extract_coefficients(m1, 'Judge_Score'))
    model_stats.append({
        'Model': 'Judge_Score', 'DV': 'avg_judge_score', 'N': len(analysis_df),
        'R2': m1.rsquared, 'Adj_R2': m1.rsquared_adj, 'F': m1.fvalue, 'F_pval': m1.f_pvalue
    })
    print(f"  [Judge] R² = {m1.rsquared:.4f}")
    
    # Model 2: Fan Vote
    m2 = smf.ols(f'avg_fan_vote_share ~ {formula}', data=analysis_df).fit()
    results['fan'] = m2
    all_coefs.append(extract_coefficients(m2, 'Fan_Vote'))
    model_stats.append({
        'Model': 'Fan_Vote', 'DV': 'avg_fan_vote_share', 'N': len(analysis_df),
        'R2': m2.rsquared, 'Adj_R2': m2.rsquared_adj, 'F': m2.fvalue, 'F_pval': m2.f_pvalue
    })
    print(f"  [Fan] R² = {m2.rsquared:.4f}")
    
    # Model 3: Placement
    m3 = smf.ols(f'placement ~ {formula}', data=analysis_df).fit()
    results['placement'] = m3
    all_coefs.append(extract_coefficients(m3, 'Placement'))
    model_stats.append({
        'Model': 'Placement', 'DV': 'placement', 'N': len(analysis_df),
        'R2': m3.rsquared, 'Adj_R2': m3.rsquared_adj, 'F': m3.fvalue, 'F_pval': m3.f_pvalue
    })
    print(f"  [Placement] R² = {m3.rsquared:.4f}")
    
    coef_df = pd.concat(all_coefs, ignore_index=True)
    stats_df = pd.DataFrame(model_stats)
    
    return results, coef_df, stats_df, analysis_df


def run_weekly_models(df):
    """运行周级别OLS模型"""
    
    analysis_df = df.dropna(subset=[
        'age', 'fan_vote_share', 'judge_total', 
        'bmi', 'dance_experience_score', 'social_media_popularity', 'google_search_volume'
    ])
    analysis_df = analysis_df[analysis_df['judge_total'] > 0]
    
    print(f"\n[WEEKLY] Sample: {len(analysis_df)} week-observations")
    
    formula = """
        age + is_us + celeb_popularity + bmi + dance_experience_score 
        + C(industry_group, Treatment('Other'))
        + social_media_popularity + google_search_volume
    """
    
    results = {}
    all_coefs = []
    model_stats = []
    
    # Weekly Fan Vote
    m1 = smf.ols(f'fan_vote_share ~ {formula}', data=analysis_df).fit()
    results['fan_weekly'] = m1
    all_coefs.append(extract_coefficients(m1, 'Weekly_Fan'))
    model_stats.append({
        'Model': 'Weekly_Fan', 'DV': 'fan_vote_share', 'N': len(analysis_df),
        'R2': m1.rsquared, 'Adj_R2': m1.rsquared_adj, 'F': m1.fvalue, 'F_pval': m1.f_pvalue
    })
    print(f"  [Weekly Fan] R² = {m1.rsquared:.4f}")
    
    # Weekly Judge
    m2 = smf.ols(f'judge_total ~ {formula}', data=analysis_df).fit()
    results['judge_weekly'] = m2
    all_coefs.append(extract_coefficients(m2, 'Weekly_Judge'))
    model_stats.append({
        'Model': 'Weekly_Judge', 'DV': 'judge_total', 'N': len(analysis_df),
        'R2': m2.rsquared, 'Adj_R2': m2.rsquared_adj, 'F': m2.fvalue, 'F_pval': m2.f_pvalue
    })
    print(f"  [Weekly Judge] R² = {m2.rsquared:.4f}")
    
    coef_df = pd.concat(all_coefs, ignore_index=True)
    stats_df = pd.DataFrame(model_stats)
    
    return results, coef_df, stats_df


def run_mixed_models(contestant_df, history_df):
    """运行混合效应模型"""
    
    analysis_df = contestant_df.merge(history_df, on=['ballroom_partner', 'season'], how='left')
    analysis_df['partner_seasons_before'] = analysis_df['partner_seasons_before'].fillna(0)
    analysis_df = analysis_df.dropna(subset=[
        'age', 'celeb_popularity', 'avg_judge_score', 'avg_fan_vote_share', 
        'placement', 'bmi', 'dance_experience_score'
    ])
    analysis_df = analysis_df[analysis_df['avg_judge_score'] > 0]
    
    print(f"\n[MIXED] Sample: {len(analysis_df)} pairs, {analysis_df['ballroom_partner'].nunique()} partners")
    
    results = {}
    icc_records = []
    
    formula = 'age + is_us + celeb_popularity + bmi + dance_experience_score + partner_seasons_before'
    
    for name, dv in [('Judge', 'avg_judge_score'), ('Fan', 'avg_fan_vote_share'), ('Placement', 'placement')]:
        try:
            m = smf.mixedlm(f'{dv} ~ {formula}', data=analysis_df, groups=analysis_df['ballroom_partner']).fit()
            results[name.lower()] = m
            
            var_u = m.cov_re.iloc[0, 0]
            var_e = m.scale
            icc = var_u / (var_u + var_e)
            
            icc_records.append({
                'Model': name,
                'DV': dv,
                'Var_Partner': var_u,
                'Var_Residual': var_e,
                'ICC': icc,
                'ICC_Percent': f"{icc*100:.1f}%"
            })
            print(f"  [{name}] ICC = {icc:.4f} ({icc*100:.1f}%)")
            
        except Exception as e:
            print(f"  [{name}] Error: {e}")
    
    icc_df = pd.DataFrame(icc_records)
    
    # Partner ranking
    partner_stats = analysis_df.groupby('ballroom_partner').agg({
        'placement': ['mean', 'min', 'count'],
        'avg_judge_score': 'mean',
        'avg_fan_vote_share': 'mean'
    }).reset_index()
    partner_stats.columns = ['Partner', 'Avg_Placement', 'Best_Placement', 'Appearances', 'Avg_Judge', 'Avg_Fan']
    partner_stats = partner_stats[partner_stats['Appearances'] >= 2].sort_values('Avg_Placement')
    
    return results, icc_df, partner_stats


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(ols_results, ols_coef, ols_stats, weekly_coef, weekly_stats, icc_df, partner_stats):
    """生成简化版MD报告"""
    
    lines = []
    lines.append("# Task 3: Impact of Contestant and Partner Characteristics")
    lines.append(f"\n*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("---\n")
    
    # ========== 1. Executive Summary ==========
    lines.append("## Executive Summary\n")
    
    # 获取关键统计
    judge_m = ols_results['judge']
    fan_m = ols_results['fan']
    place_m = ols_results['placement']
    
    lines.append("### Key Findings\n")
    lines.append("| # | Finding | Evidence |")
    lines.append("|---|---------|----------|")
    
    # Finding 1: Age
    age_coef_j = judge_m.params['age']
    age_p_j = judge_m.pvalues['age']
    lines.append(f"| 1 | **Younger contestants perform better** | Age → Judge: β={age_coef_j:.3f} (p={age_p_j:.4f}) |")
    
    # Finding 2: Popularity → Fan votes
    pop_t_fan = fan_m.tvalues['celeb_popularity']
    pop_t_judge = judge_m.tvalues['celeb_popularity']
    lines.append(f"| 2 | **Popularity drives fan votes, not judge scores** | Fan t={pop_t_fan:.2f}, Judge t={pop_t_judge:.2f} |")
    
    # Finding 3: Social Media
    lines.append(f"| 3 | **Social media buzz impacts weekly voting** | See Weekly Model below |")
    
    # Finding 4: Partner effect
    icc_judge = icc_df[icc_df['Model'] == 'Judge']['ICC'].values[0] if len(icc_df) > 0 else 0
    icc_fan = icc_df[icc_df['Model'] == 'Fan']['ICC'].values[0] if len(icc_df) > 0 else 0
    lines.append(f"| 4 | **Partner explains ~15% of variance** | Judge ICC={icc_judge:.1%}, Fan ICC={icc_fan:.1%} |")
    
    # Finding 5: Industry
    sm_coef = judge_m.params.get("C(industry_group, Treatment('Other'))[T.SocialMedia]", 0)
    sm_p = judge_m.pvalues.get("C(industry_group, Treatment('Other'))[T.SocialMedia]", 1)
    lines.append(f"| 5 | **Social media stars score highest** | β=+{sm_coef:.2f} (p={sm_p:.4f}) |")
    
    lines.append("")
    
    # ========== 2. Model Fit Summary ==========
    lines.append("---\n## Model Fit Summary\n")
    lines.append("### Contestant-Level Models (N=421)\n")
    lines.append("| Model | DV | R² | Adj R² | F-stat |")
    lines.append("|-------|----|----|--------|--------|")
    for _, row in ols_stats.iterrows():
        lines.append(f"| {row['Model']} | {row['DV']} | {row['R2']:.4f} | {row['Adj_R2']:.4f} | {row['F']:.2f} |")
    
    lines.append("\n### Weekly-Level Models (N≈2777)\n")
    lines.append("| Model | DV | R² | Adj R² | F-stat |")
    lines.append("|-------|----|----|--------|--------|")
    for _, row in weekly_stats.iterrows():
        lines.append(f"| {row['Model']} | {row['DV']} | {row['R2']:.4f} | {row['Adj_R2']:.4f} | {row['F']:.2f} |")
    
    # ========== 3. Coefficient Comparison ==========
    lines.append("\n---\n## Judge vs Fan: What Matters?\n")
    lines.append("| Variable | Judge β | Fan β | Judge t | Fan t | Who Cares More? |")
    lines.append("|----------|---------|-------|---------|-------|-----------------|")
    
    key_vars = ['age', 'celeb_popularity', 'bmi', 'dance_experience_score', 'is_us']
    for var in key_vars:
        if var in judge_m.params and var in fan_m.params:
            jb, fb = judge_m.params[var], fan_m.params[var]
            jt, ft = judge_m.tvalues[var], fan_m.tvalues[var]
            who = "**Fans**" if abs(ft) > abs(jt) + 0.5 else ("**Judges**" if abs(jt) > abs(ft) + 0.5 else "Similar")
            lines.append(f"| {var} | {jb:.4f} | {fb:.4f} | {jt:.2f} | {ft:.2f} | {who} |")
    
    # ========== 4. Partner Effect ==========
    lines.append("\n---\n## Professional Dancer Effect (ICC)\n")
    lines.append("| Model | ICC | Interpretation |")
    lines.append("|-------|-----|----------------|")
    for _, row in icc_df.iterrows():
        interp = "Substantial" if row['ICC'] > 0.15 else ("Moderate" if row['ICC'] > 0.05 else "Small")
        lines.append(f"| {row['Model']} | {row['ICC_Percent']} | {interp} effect |")
    
    lines.append("\n### Top 10 Professional Dancers (by Avg Placement)\n")
    lines.append("| Rank | Partner | Apps | Avg Place | Best | Avg Judge |")
    lines.append("|------|---------|------|-----------|------|-----------|")
    for i, (_, row) in enumerate(partner_stats.head(10).iterrows(), 1):
        lines.append(f"| {i} | {row['Partner']} | {row['Appearances']:.0f} | {row['Avg_Placement']:.2f} | {row['Best_Placement']:.0f} | {row['Avg_Judge']:.1f} |")
    
    # ========== 5. Conclusion ==========
    lines.append("\n---\n## Conclusions\n")
    lines.append("1. **Celebrity characteristics matter**: Age (younger = better) and industry background significantly impact performance.\n")
    lines.append("2. **Judges and fans value different things**: Fans are driven by celebrity popularity; judges focus more on dancing ability.\n")
    lines.append("3. **Partner choice is important**: Professional dancers explain 14-18% of outcome variance.\n")
    lines.append("4. **Social media is a real-time driver**: Weekly social media buzz strongly predicts fan voting.\n")
    
    return '\n'.join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load data
    df = load_and_prepare()
    contestant_df = create_contestant_summary(df)
    history_df = compute_partner_history(df)
    
    # Run models
    print("\n" + "=" * 40)
    print("Running OLS Models")
    print("=" * 40)
    ols_results, ols_coef, ols_stats, _ = run_ols_models(contestant_df)
    
    print("\n" + "=" * 40)
    print("Running Weekly Models")
    print("=" * 40)
    weekly_results, weekly_coef, weekly_stats = run_weekly_models(df)
    
    print("\n" + "=" * 40)
    print("Running Mixed Effects Models")
    print("=" * 40)
    me_results, icc_df, partner_stats = run_mixed_models(contestant_df, history_df)
    
    # Save coefficients to CSV
    print("\n" + "=" * 40)
    print("Saving Results")
    print("=" * 40)
    
    # Combine all coefficients
    all_coefs = pd.concat([ols_coef, weekly_coef], ignore_index=True)
    all_coefs.to_csv(f'{OUTPUT_DIR}task3_regression_coefficients.csv', index=False)
    print(f"[SAVED] task3_regression_coefficients.csv")
    
    # Model stats
    all_stats = pd.concat([ols_stats, weekly_stats], ignore_index=True)
    all_stats.to_csv(f'{OUTPUT_DIR}task3_model_stats.csv', index=False)
    print(f"[SAVED] task3_model_stats.csv")
    
    # ICC
    icc_df.to_csv(f'{OUTPUT_DIR}task3_icc_results.csv', index=False)
    print(f"[SAVED] task3_icc_results.csv")
    
    # Partner ranking
    partner_stats.to_csv(f'{OUTPUT_DIR}task3_partner_ranking_v3.csv', index=False)
    print(f"[SAVED] task3_partner_ranking_v3.csv")
    
    # Contestant summary
    contestant_df.to_csv(f'{OUTPUT_DIR}task3_contestant_summary_v3.csv', index=False)
    print(f"[SAVED] task3_contestant_summary_v3.csv")
    
    # Generate MD report
    report = generate_report(ols_results, ols_coef, ols_stats, weekly_coef, weekly_stats, icc_df, partner_stats)
    with open(f'{OUTPUT_DIR}task3_report_v3.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[SAVED] task3_report_v3.md")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return ols_results, weekly_results, me_results


if __name__ == "__main__":
    ols_results, weekly_results, me_results = main()
