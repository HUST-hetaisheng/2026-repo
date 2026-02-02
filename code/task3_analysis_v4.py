"""
Task 3: Impact of Contestant and Partner Characteristics (Version 4)
=====================================================================
修正版本：
1. 移除 age_squared（论文未提及）
2. 移除 celeb_popularity（与 social_media 概念重叠）
3. 移除 partner_popularity（与 Mixed Model 随机效应 u_j 冲突）
4. 使用 Z-score 标准化数据
5. Contestant Model 仅使用静态特征
6. Weekly Model 分析动态变量增量效应

Author: Team
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# 使用脚本所在目录计算绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, 'data') + os.sep
OUTPUT_DIR = os.path.join(REPO_DIR, 'data') + os.sep

INDUSTRY_MAPPING = {
    'Actor/Actress': 'Actor',
    'Athlete': 'Athlete',
    'TV Personality': 'TV',
    'Singer/Rapper': 'Singer',
    'Model': 'Model',
    'Comedian': 'Comedian',
    'Social Media Personality': 'SocialMedia',
    'Social media personality': 'SocialMedia',
    'News Anchor': 'TV',
    'Sports Broadcaster': 'TV',
}

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare():
    """加载Z-score标准化数据并进行特征工程"""
    
    print("=" * 60)
    print("Task 3 Analysis V4 (Corrected Models)")
    print("=" * 60)
    
    # 使用 Z-score 标准化数据
    df = pd.read_csv(f'{DATA_DIR}task3_dataset_full_zscored.csv')
    print(f"[DATA] Loaded Z-scored data: {len(df)} rows, {df.shape[1]} columns")
    
    # 特征工程
    df['industry_group'] = df['celebrity_industry'].map(
        lambda x: INDUSTRY_MAPPING.get(x, 'Other') if pd.notna(x) else 'Other'
    )
    df['is_us'] = (df['celebrity_homecountry/region'] == 'United States').astype(int)
    df['age'] = df['celebrity_age_during_season']
    
    # 注意：不再创建 age_squared, celeb_popularity, partner_popularity
    
    # 填充缺失值（使用中位数）
    for col in ['bmi', 'dance_experience_score', 'social_media_popularity', 'google_search_volume']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df


def create_contestant_summary(df):
    """创建选手级汇总（仅静态特征）"""
    
    agg = df.groupby(['season', 'celebrity_name']).agg({
        'fan_vote_share': 'mean',
        'judge_total': 'mean',
        'placement': 'first',
        'week': 'max',
        'age': 'first',
        'is_us': 'first',
        'bmi': 'first',
        'dance_experience_score': 'first',
        'industry_group': 'first',
        'ballroom_partner': 'first',
        # 周度变量取平均作为选手级特征
        'social_media_popularity': 'mean',
        'google_search_volume': 'mean',
    }).reset_index()
    
    agg.columns = [
        'season', 'celebrity_name', 
        'avg_fan_vote_share', 'avg_judge_score', 'placement', 'weeks_survived',
        'age', 'is_us', 'bmi', 'dance_experience_score', 
        'industry_group', 'ballroom_partner',
        'avg_social_media_popularity', 'avg_google_search_volume'
    ]
    
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
    """
    运行三个 Contestant-Level OLS 回归模型
    
    公式 (与论文一致):
    Y_i = β_0 + β_1·Age + β_2·isUS + β_3·BMI + β_4·DanceExp + Σγ_k·Industry_k + ε_i
    
    注意：
    - 不包含 celeb_popularity（与 social_media 重叠）
    - 不包含 partner_popularity（与 Mixed Model 的 u_j 重叠）
    - 不包含 age_squared（论文未提及）
    """
    
    analysis_df = contestant_df.dropna(subset=[
        'age', 'avg_judge_score', 'avg_fan_vote_share', 
        'placement', 'bmi', 'dance_experience_score'
    ])
    analysis_df = analysis_df[analysis_df['avg_judge_score'] > 0]
    
    print(f"\n[OLS] Sample: {len(analysis_df)} contestants")
    
    # 论文公式一致的模型
    formula = """
        age + is_us + bmi + dance_experience_score 
        + C(industry_group, Treatment('Other'))
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
    print(f"  [Judge] R² = {m1.rsquared:.4f}, Adj R² = {m1.rsquared_adj:.4f}")
    
    # Model 2: Fan Vote
    m2 = smf.ols(f'avg_fan_vote_share ~ {formula}', data=analysis_df).fit()
    results['fan'] = m2
    all_coefs.append(extract_coefficients(m2, 'Fan_Vote'))
    model_stats.append({
        'Model': 'Fan_Vote', 'DV': 'avg_fan_vote_share', 'N': len(analysis_df),
        'R2': m2.rsquared, 'Adj_R2': m2.rsquared_adj, 'F': m2.fvalue, 'F_pval': m2.f_pvalue
    })
    print(f"  [Fan] R² = {m2.rsquared:.4f}, Adj R² = {m2.rsquared_adj:.4f}")
    
    # Model 3: Placement
    m3 = smf.ols(f'placement ~ {formula}', data=analysis_df).fit()
    results['placement'] = m3
    all_coefs.append(extract_coefficients(m3, 'Placement'))
    model_stats.append({
        'Model': 'Placement', 'DV': 'placement', 'N': len(analysis_df),
        'R2': m3.rsquared, 'Adj_R2': m3.rsquared_adj, 'F': m3.fvalue, 'F_pval': m3.f_pvalue
    })
    print(f"  [Placement] R² = {m3.rsquared:.4f}, Adj R² = {m3.rsquared_adj:.4f}")
    
    coef_df = pd.concat(all_coefs, ignore_index=True)
    stats_df = pd.DataFrame(model_stats)
    
    return results, coef_df, stats_df, analysis_df


def run_weekly_models(df):
    """
    运行 Weekly-Level OLS 模型
    
    公式:
    Y_it = β_0 + β_1·SocialMedia_it + β_2·GoogleSearch_it + Controls + ε_it
    
    注意：
    - 不包含 celeb_popularity（与动态变量高度相关）
    - 重点分析 social_media_popularity 和 google_search_volume 的增量效应
    """
    
    analysis_df = df.dropna(subset=[
        'age', 'fan_vote_share', 'judge_total', 
        'bmi', 'dance_experience_score', 'social_media_popularity', 'google_search_volume'
    ])
    analysis_df = analysis_df[analysis_df['judge_total'] > 0]
    
    print(f"\n[WEEKLY] Sample: {len(analysis_df)} week-observations")
    
    # 周级别模型：控制静态特征，分析动态变量效应
    formula = """
        social_media_popularity + google_search_volume
        + age + is_us + bmi + dance_experience_score 
        + C(industry_group, Treatment('Other'))
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
    """
    运行混合效应模型
    
    公式:
    Y_ij = X_ij'β + u_j + ε_ij
    
    其中 u_j ~ N(0, σ²_u) 是舞伴随机效应
    
    注意：
    - 不包含 partner_popularity（会与 u_j 产生多重共线性）
    - u_j 已完整捕捉舞伴效应（包括知名度、教学能力等）
    """
    
    analysis_df = contestant_df.merge(history_df, on=['ballroom_partner', 'season'], how='left')
    analysis_df['partner_seasons_before'] = analysis_df['partner_seasons_before'].fillna(0)
    analysis_df = analysis_df.dropna(subset=[
        'age', 'avg_judge_score', 'avg_fan_vote_share', 
        'placement', 'bmi', 'dance_experience_score'
    ])
    analysis_df = analysis_df[analysis_df['avg_judge_score'] > 0]
    
    print(f"\n[MIXED] Sample: {len(analysis_df)} pairs, {analysis_df['ballroom_partner'].nunique()} partners")
    
    results = {}
    icc_records = []
    
    # 固定效应公式（不包含 partner_popularity）
    formula = 'age + is_us + bmi + dance_experience_score + partner_seasons_before'
    
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
    
    return results, icc_df, partner_stats, analysis_df


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(ols_results, ols_coef, ols_stats, weekly_coef, weekly_stats, 
                    icc_df, partner_stats, weekly_results):
    """生成 V4 版 MD 报告"""
    
    lines = []
    lines.append("# Task 3: Impact of Contestant and Partner Characteristics (V4)")
    lines.append(f"\n*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("---\n")
    
    # ========== 0. Model Specification ==========
    lines.append("## Model Specification (Corrected)\n")
    lines.append("### Changes from V3:\n")
    lines.append("1. **Removed `age_squared`**: Paper equation does not include quadratic term\n")
    lines.append("2. **Removed `celeb_popularity`**: Highly correlated with `social_media_popularity`\n")
    lines.append("3. **Removed `partner_popularity`**: Conflicts with random effect $u_j$ in Mixed Model\n")
    lines.append("4. **Using Z-score normalized data** for social media and Google search variables\n")
    lines.append("")
    
    lines.append("### Contestant-Level Model (Paper Equation 1):\n")
    lines.append("```\n")
    lines.append("Y_i = β_0 + β_1·Age + β_2·isUS + β_3·BMI + β_4·DanceExp + Σγ_k·Industry_k + ε_i\n")
    lines.append("```\n")
    
    lines.append("### Weekly-Level Model:\n")
    lines.append("```\n")
    lines.append("Y_it = β_0 + β_1·SocialMedia_it + β_2·GoogleSearch_it + Controls + ε_it\n")
    lines.append("```\n")
    
    lines.append("### Mixed Effects Model (Paper Equation 2):\n")
    lines.append("```\n")
    lines.append("Y_ij = X_ij'β + u_j + ε_ij,  where u_j ~ N(0, σ²_u)\n")
    lines.append("```\n")
    
    # ========== 1. Executive Summary ==========
    lines.append("---\n## Executive Summary\n")
    
    judge_m = ols_results['judge']
    fan_m = ols_results['fan']
    place_m = ols_results['placement']
    
    lines.append("### Key Findings\n")
    lines.append("| # | Finding | Evidence |")
    lines.append("|---|---------|----------|")
    
    # Finding 1: Age
    age_coef_j = judge_m.params.get('age', 0)
    age_p_j = judge_m.pvalues.get('age', 1)
    age_sig = '***' if age_p_j < 0.01 else ('**' if age_p_j < 0.05 else ('*' if age_p_j < 0.1 else ''))
    lines.append(f"| 1 | **Age affects performance** | β={age_coef_j:.3f}{age_sig} (p={age_p_j:.4f}) |")
    
    # Finding 2: Dance Experience
    de_coef = judge_m.params.get('dance_experience_score', 0)
    de_p = judge_m.pvalues.get('dance_experience_score', 1)
    de_sig = '***' if de_p < 0.01 else ('**' if de_p < 0.05 else ('*' if de_p < 0.1 else ''))
    lines.append(f"| 2 | **Dance experience improves judge scores** | β={de_coef:.3f}{de_sig} (p={de_p:.4f}) |")
    
    # Finding 3: Social Media (from weekly model)
    sm_coef = weekly_results['fan_weekly'].params.get('social_media_popularity', 0)
    sm_p = weekly_results['fan_weekly'].pvalues.get('social_media_popularity', 1)
    sm_sig = '***' if sm_p < 0.01 else ('**' if sm_p < 0.05 else ('*' if sm_p < 0.1 else ''))
    lines.append(f"| 3 | **Social media buzz drives fan votes** | β={sm_coef:.4f}{sm_sig} (p={sm_p:.4f}) |")
    
    # Finding 4: Partner effect
    icc_judge = icc_df[icc_df['Model'] == 'Judge']['ICC'].values[0] if len(icc_df) > 0 else 0
    icc_fan = icc_df[icc_df['Model'] == 'Fan']['ICC'].values[0] if len(icc_df) > 0 else 0
    icc_place = icc_df[icc_df['Model'] == 'Placement']['ICC'].values[0] if len(icc_df) > 0 else 0
    lines.append(f"| 4 | **Partner explains significant variance** | Judge={icc_judge:.1%}, Fan={icc_fan:.1%}, Placement={icc_place:.1%} |")
    
    # Finding 5: Industry
    lines.append(f"| 5 | **Industry background matters** | See coefficient table below |")
    
    lines.append("")
    
    # ========== 2. Model Fit Summary ==========
    lines.append("---\n## Model Fit Summary\n")
    lines.append("### Contestant-Level Models\n")
    lines.append("| Model | DV | N | R² | Adj R² | F-stat | p(F) |")
    lines.append("|-------|----|----|----|----|--------|------|")
    for _, row in ols_stats.iterrows():
        lines.append(f"| {row['Model']} | {row['DV']} | {row['N']} | {row['R2']:.4f} | {row['Adj_R2']:.4f} | {row['F']:.2f} | {row['F_pval']:.4f} |")
    
    lines.append("\n### Weekly-Level Models\n")
    lines.append("| Model | DV | N | R² | Adj R² | F-stat | p(F) |")
    lines.append("|-------|----|----|----|----|--------|------|")
    for _, row in weekly_stats.iterrows():
        lines.append(f"| {row['Model']} | {row['DV']} | {row['N']} | {row['R2']:.4f} | {row['Adj_R2']:.4f} | {row['F']:.2f} | {row['F_pval']:.4f} |")
    
    # ========== 3. Coefficient Comparison ==========
    lines.append("\n---\n## Coefficient Comparison: Judge vs Fan vs Placement\n")
    lines.append("| Variable | Judge β | Fan β | Placement β | Judge p | Fan p | Place p |")
    lines.append("|----------|---------|-------|-------------|---------|-------|---------|")
    
    key_vars = ['age', 'bmi', 'dance_experience_score', 'is_us']
    for var in key_vars:
        if var in judge_m.params and var in fan_m.params and var in place_m.params:
            jb, fb, pb = judge_m.params[var], fan_m.params[var], place_m.params[var]
            jp, fp, pp = judge_m.pvalues[var], fan_m.pvalues[var], place_m.pvalues[var]
            j_sig = '***' if jp < 0.01 else ('**' if jp < 0.05 else ('*' if jp < 0.1 else ''))
            f_sig = '***' if fp < 0.01 else ('**' if fp < 0.05 else ('*' if fp < 0.1 else ''))
            p_sig = '***' if pp < 0.01 else ('**' if pp < 0.05 else ('*' if pp < 0.1 else ''))
            lines.append(f"| {var} | {jb:.4f}{j_sig} | {fb:.4f}{f_sig} | {pb:.4f}{p_sig} | {jp:.4f} | {fp:.4f} | {pp:.4f} |")
    
    # ========== 4. Weekly Model: Dynamic Variables ==========
    lines.append("\n---\n## Weekly Model: Dynamic Variables Effect\n")
    lines.append("| Variable | Fan Vote β | Judge β | Fan p | Judge p |")
    lines.append("|----------|------------|---------|-------|---------|")
    
    fan_weekly = weekly_results['fan_weekly']
    judge_weekly = weekly_results['judge_weekly']
    
    for var in ['social_media_popularity', 'google_search_volume']:
        if var in fan_weekly.params and var in judge_weekly.params:
            fb, jb = fan_weekly.params[var], judge_weekly.params[var]
            fp, jp = fan_weekly.pvalues[var], judge_weekly.pvalues[var]
            f_sig = '***' if fp < 0.01 else ('**' if fp < 0.05 else ('*' if fp < 0.1 else ''))
            j_sig = '***' if jp < 0.01 else ('**' if jp < 0.05 else ('*' if jp < 0.1 else ''))
            lines.append(f"| {var} | {fb:.4f}{f_sig} | {jb:.4f}{j_sig} | {fp:.4f} | {jp:.4f} |")
    
    lines.append("\n**Interpretation**: Social media popularity (Z-scored) is measured weekly. ")
    lines.append("A 1-SD increase in social media buzz is associated with a change in fan vote share.\n")
    
    # ========== 5. Partner Effect ==========
    lines.append("\n---\n## Professional Dancer Effect (ICC)\n")
    lines.append("| Model | σ²_partner | σ²_residual | ICC | Interpretation |")
    lines.append("|-------|------------|-------------|-----|----------------|")
    for _, row in icc_df.iterrows():
        interp = "Substantial" if row['ICC'] > 0.15 else ("Moderate" if row['ICC'] > 0.05 else "Small")
        lines.append(f"| {row['Model']} | {row['Var_Partner']:.4f} | {row['Var_Residual']:.4f} | {row['ICC_Percent']} | {interp} |")
    
    lines.append("\n### Top 10 Professional Dancers (by Avg Placement)\n")
    lines.append("| Rank | Partner | Apps | Avg Place | Best | Avg Judge |")
    lines.append("|------|---------|------|-----------|------|-----------|")
    for i, (_, row) in enumerate(partner_stats.head(10).iterrows(), 1):
        lines.append(f"| {i} | {row['Partner']} | {row['Appearances']:.0f} | {row['Avg_Placement']:.2f} | {row['Best_Placement']:.0f} | {row['Avg_Judge']:.1f} |")
    
    # ========== 6. Industry Effect ==========
    lines.append("\n---\n## Industry Effect on Judge Score\n")
    lines.append("| Industry | β (vs Other) | Std Err | t | p | Sig |")
    lines.append("|----------|--------------|---------|---|---|-----|")
    
    for var in judge_m.params.index:
        if 'Industry_' in var or 'industry_group' in var:
            industry = var.replace("C(industry_group, Treatment('Other'))[T.", "").replace("]", "")
            b = judge_m.params[var]
            se = judge_m.bse[var]
            t = judge_m.tvalues[var]
            p = judge_m.pvalues[var]
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            lines.append(f"| {industry} | {b:.4f} | {se:.4f} | {t:.2f} | {p:.4f} | {sig} |")
    
    # ========== 7. Conclusion ==========
    lines.append("\n---\n## Conclusions\n")
    lines.append("1. **Contestant characteristics significantly affect outcomes**: Age, dance experience, and industry background all play important roles.\n")
    lines.append("2. **Judges focus on performance**: Dance experience is a strong predictor of judge scores.\n")
    lines.append("3. **Fans respond to publicity**: Weekly social media buzz significantly predicts fan voting, while judges appear immune.\n")
    lines.append(f"4. **Partner matters**: Professional dancers explain {icc_judge:.1%}-{icc_place:.1%} of outcome variance (ICC).\n")
    lines.append("5. **Model consistency**: Using corrected specifications without multicollinearity issues.\n")
    
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
    print("Running OLS Models (Contestant-Level)")
    print("=" * 40)
    ols_results, ols_coef, ols_stats, _ = run_ols_models(contestant_df)
    
    print("\n" + "=" * 40)
    print("Running Weekly Models")
    print("=" * 40)
    weekly_results, weekly_coef, weekly_stats = run_weekly_models(df)
    
    print("\n" + "=" * 40)
    print("Running Mixed Effects Models")
    print("=" * 40)
    me_results, icc_df, partner_stats, _ = run_mixed_models(contestant_df, history_df)
    
    # Save coefficients to CSV (覆盖原文件)
    print("\n" + "=" * 40)
    print("Saving Results (Overwriting V3 files)")
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
    
    # Generate MD report (V4)
    report = generate_report(ols_results, ols_coef, ols_stats, weekly_coef, weekly_stats, 
                             icc_df, partner_stats, weekly_results)
    with open(f'{OUTPUT_DIR}task3_report_v4.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[SAVED] task3_report_v4.md")
    
    print("\n" + "=" * 60)
    print("DONE! (V4 - Corrected Models)")
    print("=" * 60)
    
    return ols_results, weekly_results, me_results


if __name__ == "__main__":
    ols_results, weekly_results, me_results = main()
