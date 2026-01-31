"""
Task 3: Impact of Contestant and Partner Characteristics
==========================================================
2026 MCM Problem C - DWTS Analysis

This script performs:
1. Data preparation and feature engineering
2. Module A: Celebrity characteristics analysis (OLS)
3. Module B: Professional dancer analysis (Mixed Effects Model)
4. Effect comparison: Judge scores vs Fan votes
5. Output results to markdown file

Author: Team
Date: 2026-01
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load all data sources and prepare for analysis"""
    
    # Load data files
    pop_df = pd.read_csv('../data/2026_MCM_Problem_C_Data_Cleaned添加人气后.csv')
    fan_df = pd.read_csv('../data/fan_vote_results_final.csv')
    
    print(f"[DATA] Loaded {len(pop_df)} contestants from pop_df")
    print(f"[DATA] Loaded {len(fan_df)} weekly records from fan_df")
    
    return pop_df, fan_df


def categorize_region(country):
    """Categorize country/region into groups"""
    if pd.isna(country) or country == '':
        return 'Unknown'
    elif country == 'United States':
        return 'US'
    elif country in ['England', 'United Kingdom', 'Ireland', 'Scotland', 
                     'Australia', 'New Zealand', 'Canada']:
        return 'English_Speaking'
    else:
        return 'Other'


def extract_industry(results_str, celebrity_name):
    """
    Extract industry category from available data
    For this analysis, we'll use a simplified categorization based on 
    common patterns in celebrity names/results
    """
    # This is a simplified approach - in real analysis, 
    # industry data should be explicitly provided
    return 'Unknown'


def compute_weekly_judge_scores(row, max_weeks=11):
    """Compute weekly total judge scores for a contestant"""
    weekly_totals = []
    
    for w in range(1, max_weeks + 1):
        week_scores = []
        for j in range(1, 5):
            col_name = f'week{w}_judge{j}_score'
            if col_name in row.index:
                score = row[col_name]
                if pd.notna(score) and score > 0:
                    week_scores.append(score)
        
        if week_scores:
            weekly_totals.append(sum(week_scores))
    
    return weekly_totals


def create_contestant_summary(pop_df, fan_df):
    """Create contestant-level summary statistics"""
    
    summary_records = []
    
    for idx, row in pop_df.iterrows():
        # Compute weekly judge scores
        weekly_totals = compute_weekly_judge_scores(row)
        
        avg_judge_score = np.mean(weekly_totals) if weekly_totals else 0
        max_judge_score = np.max(weekly_totals) if weekly_totals else 0
        weeks_survived = len(weekly_totals)
        
        # Extract placement as numeric
        placement = row['placement']
        if pd.isna(placement):
            placement_num = np.nan
        elif isinstance(placement, (int, float)):
            placement_num = placement
        else:
            placement_num = placement  # Already numeric in cleaned data
        
        summary_records.append({
            'celebrity_name': row['celebrity_name'],
            'ballroom_partner': row['ballroom_partner'],
            'season': row['season'],
            'placement': placement_num,
            'avg_judge_score': avg_judge_score,
            'max_judge_score': max_judge_score,
            'weeks_survived': weeks_survived,
            'age': row['celebrity_age_during_season'],
            'country': row['celebrity_homecountry/region'],
            'celeb_popularity': row['Celebrity_Average_Popularity_Score'],
            'partner_popularity': row['ballroom_partner_Average_Popularity_Score']
        })
    
    summary_df = pd.DataFrame(summary_records)
    
    # Add derived features
    summary_df['is_us'] = (summary_df['country'] == 'United States').astype(int)
    summary_df['region_group'] = summary_df['country'].apply(categorize_region)
    summary_df['age_squared'] = summary_df['age'] ** 2
    summary_df['celeb_pop_log'] = np.log(summary_df['celeb_popularity'] + 1)
    summary_df['partner_pop_log'] = np.log(summary_df['partner_popularity'] + 1)
    
    # Merge fan vote summary
    fan_summary = fan_df.groupby(['season', 'celebrity_name']).agg({
        'fan_vote_share': 'mean',
        'cv': 'mean'
    }).reset_index()
    fan_summary.columns = ['season', 'celebrity_name', 'avg_fan_vote_share', 'avg_cv']
    
    summary_df = summary_df.merge(fan_summary, on=['season', 'celebrity_name'], how='left')
    
    print(f"[SUMMARY] Created summary for {len(summary_df)} contestants")
    
    return summary_df


def compute_partner_history(pop_df):
    """
    Compute partner historical performance statistics
    For each (partner, season), compute stats from PRIOR seasons only
    """
    
    # Get partner-season placements
    partner_seasons = pop_df.groupby(['ballroom_partner', 'season']).agg({
        'placement': 'first',
        'celebrity_name': 'first'
    }).reset_index()
    
    history_records = []
    
    for partner in partner_seasons['ballroom_partner'].unique():
        partner_data = partner_seasons[partner_seasons['ballroom_partner'] == partner].sort_values('season')
        
        placements_before = []
        
        for _, row in partner_data.iterrows():
            current_season = row['season']
            current_placement = row['placement']
            
            history_records.append({
                'ballroom_partner': partner,
                'season': current_season,
                'partner_seasons_before': len(placements_before),
                'partner_avg_placement_before': np.mean(placements_before) if placements_before else np.nan,
                'partner_best_placement_before': min(placements_before) if placements_before else np.nan,
                'partner_wins_before': sum(1 for p in placements_before if p == 1)
            })
            
            # Add current placement to history for next iteration
            if pd.notna(current_placement):
                placements_before.append(current_placement)
    
    history_df = pd.DataFrame(history_records)
    
    # Fill NaN for first-time partners
    history_df['partner_seasons_before'] = history_df['partner_seasons_before'].fillna(0)
    history_df['partner_wins_before'] = history_df['partner_wins_before'].fillna(0)
    
    print(f"[HISTORY] Computed history for {history_df['ballroom_partner'].nunique()} unique partners")
    
    return history_df


# ============================================================================
# PART 2: MODULE A - CELEBRITY CHARACTERISTICS ANALYSIS
# ============================================================================

def run_celebrity_ols_analysis(df, output_lines):
    """
    Run OLS regression for celebrity characteristics
    
    Models:
    1. Judge Score Model
    2. Fan Vote Model  
    3. Placement Model
    """
    
    output_lines.append("\n## Module A: Celebrity Characteristics Analysis (OLS)")
    output_lines.append("\n### Model Specification")
    output_lines.append("""
$$Y_i = \\beta_0 + \\beta_1 \\cdot Age_i + \\beta_2 \\cdot Age_i^2 + \\beta_3 \\cdot isUS_i + \\beta_4 \\cdot \\log(Popularity_i + 1) + \\epsilon_i$$
""")
    
    # Prepare data - remove rows with missing values
    analysis_df = df.dropna(subset=['age', 'celeb_pop_log', 'avg_judge_score', 'avg_fan_vote_share', 'placement'])
    analysis_df = analysis_df[analysis_df['avg_judge_score'] > 0]
    
    print(f"\n[MODULE A] Analyzing {len(analysis_df)} contestants with complete data")
    output_lines.append(f"\n**Sample Size:** {len(analysis_df)} contestants with complete data\n")
    
    results_summary = {}
    
    # ---- Model 1: Judge Score ----
    output_lines.append("\n### Model 1: Average Judge Score")
    
    try:
        model_judge = smf.ols(
            'avg_judge_score ~ age + age_squared + is_us + celeb_pop_log + partner_pop_log',
            data=analysis_df
        ).fit()
        
        results_summary['judge'] = model_judge
        
        output_lines.append(f"\n**R-squared:** {model_judge.rsquared:.4f}")
        output_lines.append(f"**Adjusted R-squared:** {model_judge.rsquared_adj:.4f}")
        output_lines.append(f"**F-statistic:** {model_judge.fvalue:.2f} (p = {model_judge.f_pvalue:.4f})")
        
        output_lines.append("\n| Variable | Coefficient | Std Error | t-value | p-value | Significance |")
        output_lines.append("|----------|-------------|-----------|---------|---------|--------------|")
        
        for var in model_judge.params.index:
            coef = model_judge.params[var]
            se = model_judge.bse[var]
            t = model_judge.tvalues[var]
            p = model_judge.pvalues[var]
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            output_lines.append(f"| {var} | {coef:.4f} | {se:.4f} | {t:.2f} | {p:.4f} | {sig} |")
        
        print(f"  [Judge Score] R² = {model_judge.rsquared:.4f}")
        
    except Exception as e:
        output_lines.append(f"\n**Error in Judge Score Model:** {str(e)}")
        print(f"  [Judge Score] Error: {e}")
    
    # ---- Model 2: Fan Vote Share ----
    output_lines.append("\n### Model 2: Average Fan Vote Share")
    
    try:
        model_fan = smf.ols(
            'avg_fan_vote_share ~ age + age_squared + is_us + celeb_pop_log + partner_pop_log',
            data=analysis_df
        ).fit()
        
        results_summary['fan'] = model_fan
        
        output_lines.append(f"\n**R-squared:** {model_fan.rsquared:.4f}")
        output_lines.append(f"**Adjusted R-squared:** {model_fan.rsquared_adj:.4f}")
        output_lines.append(f"**F-statistic:** {model_fan.fvalue:.2f} (p = {model_fan.f_pvalue:.4f})")
        
        output_lines.append("\n| Variable | Coefficient | Std Error | t-value | p-value | Significance |")
        output_lines.append("|----------|-------------|-----------|---------|---------|--------------|")
        
        for var in model_fan.params.index:
            coef = model_fan.params[var]
            se = model_fan.bse[var]
            t = model_fan.tvalues[var]
            p = model_fan.pvalues[var]
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            output_lines.append(f"| {var} | {coef:.4f} | {se:.4f} | {t:.2f} | {p:.4f} | {sig} |")
        
        print(f"  [Fan Vote] R² = {model_fan.rsquared:.4f}")
        
    except Exception as e:
        output_lines.append(f"\n**Error in Fan Vote Model:** {str(e)}")
        print(f"  [Fan Vote] Error: {e}")
    
    # ---- Model 3: Placement ----
    output_lines.append("\n### Model 3: Final Placement (Lower is Better)")
    
    try:
        model_place = smf.ols(
            'placement ~ age + age_squared + is_us + celeb_pop_log + partner_pop_log',
            data=analysis_df
        ).fit()
        
        results_summary['placement'] = model_place
        
        output_lines.append(f"\n**R-squared:** {model_place.rsquared:.4f}")
        output_lines.append(f"**Adjusted R-squared:** {model_place.rsquared_adj:.4f}")
        output_lines.append(f"**F-statistic:** {model_place.fvalue:.2f} (p = {model_place.f_pvalue:.4f})")
        
        output_lines.append("\n| Variable | Coefficient | Std Error | t-value | p-value | Significance |")
        output_lines.append("|----------|-------------|-----------|---------|---------|--------------|")
        
        for var in model_place.params.index:
            coef = model_place.params[var]
            se = model_place.bse[var]
            t = model_place.tvalues[var]
            p = model_place.pvalues[var]
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            output_lines.append(f"| {var} | {coef:.4f} | {se:.4f} | {t:.2f} | {p:.4f} | {sig} |")
        
        print(f"  [Placement] R² = {model_place.rsquared:.4f}")
        
    except Exception as e:
        output_lines.append(f"\n**Error in Placement Model:** {str(e)}")
        print(f"  [Placement] Error: {e}")
    
    return results_summary


def compute_age_optimal(model_result):
    """Compute optimal age from quadratic model"""
    try:
        beta1 = model_result.params['age']
        beta2 = model_result.params['age_squared']
        if beta2 != 0:
            optimal_age = -beta1 / (2 * beta2)
            return optimal_age
    except:
        pass
    return None


# ============================================================================
# PART 3: MODULE B - PROFESSIONAL DANCER ANALYSIS
# ============================================================================

def run_partner_analysis(df, history_df, output_lines):
    """
    Run Mixed Effects Model for partner analysis
    
    Model: Y_ij = β0 + X_ij*β + u_j + ε_ij
    where u_j is the random effect for partner j
    """
    
    output_lines.append("\n---\n## Module B: Professional Dancer Analysis (Mixed Effects Model)")
    
    output_lines.append("\n### Model Specification")
    output_lines.append("""
$$Y_{ij} = \\beta_0 + \\beta_1 \\cdot Age_{ij} + \\beta_2 \\cdot isUS_{ij} + \\beta_3 \\cdot \\log(Pop_{ij}+1) + \\beta_4 \\cdot PartnerExp_j + u_j + \\epsilon_{ij}$$

Where:
- $i$ indexes contestants, $j$ indexes professional dancers
- $u_j \\sim N(0, \\sigma^2_u)$ is the random effect for dancer $j$
- $\\sigma^2_u / (\\sigma^2_u + \\sigma^2_\\epsilon)$ is the Intraclass Correlation (ICC)
""")
    
    # Merge history data
    analysis_df = df.merge(history_df, on=['ballroom_partner', 'season'], how='left')
    analysis_df['partner_seasons_before'] = analysis_df['partner_seasons_before'].fillna(0)
    
    # Remove missing values
    analysis_df = analysis_df.dropna(subset=['age', 'celeb_pop_log', 'avg_judge_score', 'avg_fan_vote_share', 'placement'])
    analysis_df = analysis_df[analysis_df['avg_judge_score'] > 0]
    
    # Count partners with multiple appearances
    partner_counts = analysis_df['ballroom_partner'].value_counts()
    multi_appearance = (partner_counts > 1).sum()
    
    print(f"\n[MODULE B] Analyzing {len(analysis_df)} contestant-partner pairs")
    print(f"  [INFO] {analysis_df['ballroom_partner'].nunique()} unique partners, {multi_appearance} with 2+ appearances")
    
    output_lines.append(f"\n**Sample Size:** {len(analysis_df)} contestant-partner pairs")
    output_lines.append(f"**Unique Partners:** {analysis_df['ballroom_partner'].nunique()}")
    output_lines.append(f"**Partners with 2+ appearances:** {multi_appearance}")
    
    results_summary = {}
    
    # ---- Mixed Effects Model 1: Judge Score ----
    output_lines.append("\n### Mixed Model 1: Average Judge Score")
    
    try:
        # Use statsmodels MixedLM
        model_judge_me = smf.mixedlm(
            'avg_judge_score ~ age + is_us + celeb_pop_log + partner_seasons_before',
            data=analysis_df,
            groups=analysis_df['ballroom_partner']
        ).fit()
        
        results_summary['judge_me'] = model_judge_me
        
        # Extract variance components
        var_u = model_judge_me.cov_re.iloc[0, 0]  # Random effect variance
        var_e = model_judge_me.scale  # Residual variance
        icc = var_u / (var_u + var_e) if (var_u + var_e) > 0 else 0
        
        output_lines.append(f"\n**Convergence:** {'Yes' if model_judge_me.converged else 'No'}")
        output_lines.append(f"**Log-Likelihood:** {model_judge_me.llf:.2f}")
        output_lines.append(f"**Random Effect Variance (σ²_u):** {var_u:.4f}")
        output_lines.append(f"**Residual Variance (σ²_ε):** {var_e:.4f}")
        output_lines.append(f"**Intraclass Correlation (ICC):** {icc:.4f} ({icc*100:.1f}%)")
        
        output_lines.append("\n**Fixed Effects:**\n")
        output_lines.append("| Variable | Coefficient | Std Error | z-value | p-value | Significance |")
        output_lines.append("|----------|-------------|-----------|---------|---------|--------------|")
        
        for var in model_judge_me.fe_params.index:
            coef = model_judge_me.fe_params[var]
            se = model_judge_me.bse_fe[var]
            z = coef / se if se > 0 else 0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            output_lines.append(f"| {var} | {coef:.4f} | {se:.4f} | {z:.2f} | {p:.4f} | {sig} |")
        
        output_lines.append(f"\n**Interpretation:** {icc*100:.1f}% of the variance in judge scores is attributable to differences between professional dancers (partner effect).")
        
        print(f"  [Judge ME] ICC = {icc:.4f}, Var_u = {var_u:.4f}")
        
    except Exception as e:
        output_lines.append(f"\n**Error in Judge Mixed Model:** {str(e)}")
        print(f"  [Judge ME] Error: {e}")
    
    # ---- Mixed Effects Model 2: Fan Vote ----
    output_lines.append("\n### Mixed Model 2: Average Fan Vote Share")
    
    try:
        model_fan_me = smf.mixedlm(
            'avg_fan_vote_share ~ age + is_us + celeb_pop_log + partner_seasons_before',
            data=analysis_df,
            groups=analysis_df['ballroom_partner']
        ).fit()
        
        results_summary['fan_me'] = model_fan_me
        
        var_u = model_fan_me.cov_re.iloc[0, 0]
        var_e = model_fan_me.scale
        icc = var_u / (var_u + var_e) if (var_u + var_e) > 0 else 0
        
        output_lines.append(f"\n**Convergence:** {'Yes' if model_fan_me.converged else 'No'}")
        output_lines.append(f"**Log-Likelihood:** {model_fan_me.llf:.2f}")
        output_lines.append(f"**Random Effect Variance (σ²_u):** {var_u:.4f}")
        output_lines.append(f"**Residual Variance (σ²_ε):** {var_e:.4f}")
        output_lines.append(f"**Intraclass Correlation (ICC):** {icc:.4f} ({icc*100:.1f}%)")
        
        output_lines.append("\n**Fixed Effects:**\n")
        output_lines.append("| Variable | Coefficient | Std Error | z-value | p-value | Significance |")
        output_lines.append("|----------|-------------|-----------|---------|---------|--------------|")
        
        for var in model_fan_me.fe_params.index:
            coef = model_fan_me.fe_params[var]
            se = model_fan_me.bse_fe[var]
            z = coef / se if se > 0 else 0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            output_lines.append(f"| {var} | {coef:.4f} | {se:.4f} | {z:.2f} | {p:.4f} | {sig} |")
        
        output_lines.append(f"\n**Interpretation:** {icc*100:.1f}% of the variance in fan votes is attributable to differences between professional dancers.")
        
        print(f"  [Fan ME] ICC = {icc:.4f}, Var_u = {var_u:.4f}")
        
    except Exception as e:
        output_lines.append(f"\n**Error in Fan Mixed Model:** {str(e)}")
        print(f"  [Fan ME] Error: {e}")
    
    # ---- Partner Ranking Table ----
    output_lines.append("\n### Partner Performance Ranking")
    output_lines.append("\nTop 10 professional dancers by average partner placement:\n")
    
    # Compute partner statistics
    partner_stats = analysis_df.groupby('ballroom_partner').agg({
        'placement': ['mean', 'min', 'count'],
        'avg_judge_score': 'mean',
        'avg_fan_vote_share': 'mean'
    }).reset_index()
    partner_stats.columns = ['Partner', 'Avg_Placement', 'Best_Placement', 'Appearances', 'Avg_Judge', 'Avg_Fan']
    partner_stats = partner_stats[partner_stats['Appearances'] >= 2]  # At least 2 appearances
    partner_stats = partner_stats.sort_values('Avg_Placement')
    
    output_lines.append("| Rank | Partner | Appearances | Avg Placement | Best Placement | Avg Judge | Avg Fan Vote |")
    output_lines.append("|------|---------|-------------|---------------|----------------|-----------|--------------|")
    
    for rank, (_, row) in enumerate(partner_stats.head(10).iterrows(), 1):
        output_lines.append(f"| {rank} | {row['Partner']} | {row['Appearances']:.0f} | {row['Avg_Placement']:.2f} | {row['Best_Placement']:.0f} | {row['Avg_Judge']:.1f} | {row['Avg_Fan']:.3f} |")
    
    print(f"  [Partner Rank] Top partner: {partner_stats.iloc[0]['Partner']} with avg placement {partner_stats.iloc[0]['Avg_Placement']:.2f}")
    
    return results_summary, analysis_df


# ============================================================================
# PART 4: EFFECT COMPARISON - JUDGE vs FAN
# ============================================================================

def compare_judge_vs_fan_effects(ols_results, me_results, df, output_lines):
    """
    Compare how characteristics affect judge scores vs fan votes differently
    """
    
    output_lines.append("\n---\n## Module C: Effect Comparison - Judge Scores vs Fan Votes")
    
    output_lines.append("\n### Key Question: Do characteristics impact judges and fans differently?")
    
    # ---- Standardized Coefficient Comparison ----
    output_lines.append("\n### Standardized Coefficient Comparison (OLS Models)")
    
    if 'judge' in ols_results and 'fan' in ols_results:
        judge_model = ols_results['judge']
        fan_model = ols_results['fan']
        
        # Get standardized coefficients (beta weights)
        output_lines.append("\n| Variable | Judge Score (β) | Fan Vote (β) | Difference | Interpretation |")
        output_lines.append("|----------|-----------------|--------------|------------|----------------|")
        
        common_vars = set(judge_model.params.index) & set(fan_model.params.index)
        
        for var in ['age', 'is_us', 'celeb_pop_log', 'partner_pop_log']:
            if var in common_vars:
                judge_coef = judge_model.params[var]
                fan_coef = fan_model.params[var]
                
                # Standardize (rough approximation using t-values as proxy)
                judge_t = judge_model.tvalues[var]
                fan_t = fan_model.tvalues[var]
                
                diff = abs(judge_t) - abs(fan_t)
                
                if var == 'celeb_pop_log':
                    interp = "Popularity: " + ("Higher fan impact" if fan_t > judge_t else "Higher judge impact")
                elif var == 'is_us':
                    interp = "US Origin: " + ("Higher fan impact" if fan_t > judge_t else "Higher judge impact")
                elif var == 'age':
                    interp = "Age: " + ("Higher fan impact" if abs(fan_t) > abs(judge_t) else "Higher judge impact")
                else:
                    interp = ""
                
                output_lines.append(f"| {var} | {judge_t:.2f} | {fan_t:.2f} | {diff:.2f} | {interp} |")
        
        output_lines.append("\n*Note: Values are t-statistics; larger absolute values indicate stronger effects.*")
    
    # ---- ICC Comparison ----
    output_lines.append("\n### Partner Effect Comparison (ICC from Mixed Models)")
    
    if 'judge_me' in me_results and 'fan_me' in me_results:
        judge_me = me_results['judge_me']
        fan_me = me_results['fan_me']
        
        judge_var_u = judge_me.cov_re.iloc[0, 0]
        judge_var_e = judge_me.scale
        judge_icc = judge_var_u / (judge_var_u + judge_var_e)
        
        fan_var_u = fan_me.cov_re.iloc[0, 0]
        fan_var_e = fan_me.scale
        fan_icc = fan_var_u / (fan_var_u + fan_var_e)
        
        output_lines.append("\n| Metric | Judge Score Model | Fan Vote Model | Interpretation |")
        output_lines.append("|--------|-------------------|----------------|----------------|")
        output_lines.append(f"| ICC | {judge_icc:.4f} ({judge_icc*100:.1f}%) | {fan_icc:.4f} ({fan_icc*100:.1f}%) | {'Partner matters more for judges' if judge_icc > fan_icc else 'Partner matters more for fans'} |")
        output_lines.append(f"| σ²_partner | {judge_var_u:.4f} | {fan_var_u:.4f} | Variance from partner differences |")
        output_lines.append(f"| σ²_residual | {judge_var_e:.4f} | {fan_var_e:.4f} | Unexplained variance |")
        
        if judge_icc > fan_icc:
            output_lines.append(f"\n**Key Finding:** Professional dancer choice explains **{judge_icc*100:.1f}%** of judge score variance but only **{fan_icc*100:.1f}%** of fan vote variance. This suggests judges are more sensitive to dancing quality (influenced by partner skill) while fans vote based on other factors (celebrity appeal, storyline, etc.).")
        else:
            output_lines.append(f"\n**Key Finding:** Professional dancer choice explains **{fan_icc*100:.1f}%** of fan vote variance vs **{judge_icc*100:.1f}%** for judges. This suggests fans are influenced by partner popularity/familiarity.")
    
    # ---- Correlation Analysis ----
    output_lines.append("\n### Correlation: Judge Scores vs Fan Votes")
    
    valid_df = df.dropna(subset=['avg_judge_score', 'avg_fan_vote_share'])
    valid_df = valid_df[valid_df['avg_judge_score'] > 0]
    
    if len(valid_df) > 10:
        corr, p_value = stats.pearsonr(valid_df['avg_judge_score'], valid_df['avg_fan_vote_share'])
        
        output_lines.append(f"\n- **Pearson Correlation:** r = {corr:.4f}")
        output_lines.append(f"- **p-value:** {p_value:.4e}")
        output_lines.append(f"- **Sample Size:** n = {len(valid_df)}")
        
        if corr > 0.7:
            output_lines.append(f"\n**Interpretation:** Strong positive correlation (r = {corr:.2f}) suggests judges and fans generally agree on performance quality.")
        elif corr > 0.4:
            output_lines.append(f"\n**Interpretation:** Moderate correlation (r = {corr:.2f}) suggests some agreement between judges and fans, but also systematic differences in what they value.")
        else:
            output_lines.append(f"\n**Interpretation:** Weak correlation (r = {corr:.2f}) suggests judges and fans use very different criteria when evaluating contestants.")
        
        print(f"\n[COMPARE] Judge-Fan correlation: r = {corr:.4f}")


# ============================================================================
# PART 5: KEY FINDINGS SUMMARY
# ============================================================================

def generate_key_findings(ols_results, me_results, df, output_lines):
    """Generate summary of key findings"""
    
    output_lines.append("\n---\n## Key Findings Summary")
    
    output_lines.append("\n### Finding 1: Age Effect")
    if 'judge' in ols_results:
        age_coef = ols_results['judge'].params.get('age', 0)
        age_sq_coef = ols_results['judge'].params.get('age_squared', 0)
        age_p = ols_results['judge'].pvalues.get('age', 1)
        age_sq_p = ols_results['judge'].pvalues.get('age_squared', 1)
        
        output_lines.append(f"- Age coefficient: β₁ = {age_coef:.4f} (p = {age_p:.4f})")
        output_lines.append(f"- Age² coefficient: β₂ = {age_sq_coef:.6f} (p = {age_sq_p:.4f})")
        
        if age_sq_coef != 0:
            optimal_age = -age_coef / (2 * age_sq_coef)
            if 20 < optimal_age < 70:
                output_lines.append(f"- Age shows an **inverted-U relationship** with judge scores")
                output_lines.append(f"- Optimal age for judge scores: approximately **{optimal_age:.0f} years old**")
                output_lines.append(f"- Too young or too old contestants receive lower scores")
            else:
                output_lines.append(f"- Age effect is **monotonic** (optimal age = {optimal_age:.0f} outside typical range)")
                if age_coef < 0:
                    output_lines.append(f"- **Younger contestants tend to receive higher judge scores**")
                else:
                    output_lines.append(f"- **Older contestants tend to receive higher judge scores**")
    
    output_lines.append("\n### Finding 2: Popularity Effect")
    if 'judge' in ols_results and 'fan' in ols_results:
        judge_pop_t = ols_results['judge'].tvalues.get('celeb_pop_log', 0)
        fan_pop_t = ols_results['fan'].tvalues.get('celeb_pop_log', 0)
        
        output_lines.append(f"- Celebrity popularity effect on judge scores: t = {judge_pop_t:.2f}")
        output_lines.append(f"- Celebrity popularity effect on fan votes: t = {fan_pop_t:.2f}")
        
        if abs(fan_pop_t) > abs(judge_pop_t):
            output_lines.append("- **Fans are more influenced by celebrity popularity than judges**")
        else:
            output_lines.append("- **Judges are more influenced by celebrity status than fans**")
    
    output_lines.append("\n### Finding 3: US vs International Contestants")
    if 'fan' in ols_results:
        us_coef = ols_results['fan'].params.get('is_us', 0)
        us_p = ols_results['fan'].pvalues.get('is_us', 1)
        
        if us_p < 0.1:
            direction = "advantage" if us_coef > 0 else "disadvantage"
            output_lines.append(f"- US-born contestants have a significant **{direction}** in fan voting (p = {us_p:.4f})")
        else:
            output_lines.append(f"- No significant difference between US and international contestants (p = {us_p:.4f})")
    
    output_lines.append("\n### Finding 4: Professional Dancer Effect")
    if 'judge_me' in me_results:
        judge_me = me_results['judge_me']
        var_u = judge_me.cov_re.iloc[0, 0]
        var_e = judge_me.scale
        icc = var_u / (var_u + var_e)
        
        output_lines.append(f"- Professional dancer explains **{icc*100:.1f}%** of variance in judge scores (ICC)")
        
        if icc > 0.15:
            output_lines.append("- This is a **substantial** partner effect - choice of professional dancer matters significantly")
        elif icc > 0.05:
            output_lines.append("- This is a **moderate** partner effect")
        else:
            output_lines.append("- This is a **small** partner effect - celebrity characteristics matter more")
    
    output_lines.append("\n### Finding 5: Partner Experience")
    if 'judge_me' in me_results:
        exp_coef = me_results['judge_me'].fe_params.get('partner_seasons_before', None)
        if exp_coef is not None:
            exp_se = me_results['judge_me'].bse_fe.get('partner_seasons_before', 0)
            exp_z = exp_coef / exp_se if exp_se > 0 else 0
            exp_p = 2 * (1 - stats.norm.cdf(abs(exp_z)))
            
            if exp_p < 0.1:
                direction = "positive" if exp_coef > 0 else "negative"
                output_lines.append(f"- Partner experience has a **{direction}** effect on performance (coef = {exp_coef:.3f}, p = {exp_p:.4f})")
                output_lines.append(f"- Each additional season of partner experience {'increases' if exp_coef > 0 else 'decreases'} judge scores by {abs(exp_coef):.2f} points on average")
            else:
                output_lines.append(f"- Partner experience effect is not statistically significant (p = {exp_p:.4f})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Task 3: Impact of Contestant and Partner Characteristics")
    print("=" * 60)
    
    # Initialize output
    output_lines = []
    output_lines.append("# Task 3 Analysis Results")
    output_lines.append("# Impact of Contestant and Partner Characteristics")
    output_lines.append(f"\n*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")
    output_lines.append("---")
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    pop_df, fan_df = load_and_prepare_data()
    
    # Step 2: Create contestant summary
    print("\n[STEP 2] Creating contestant summary...")
    contestant_df = create_contestant_summary(pop_df, fan_df)
    
    # Step 3: Compute partner history
    print("\n[STEP 3] Computing partner history...")
    history_df = compute_partner_history(pop_df)
    
    # Step 4: Run Module A - Celebrity OLS Analysis
    print("\n[STEP 4] Running Module A: Celebrity Characteristics Analysis...")
    ols_results = run_celebrity_ols_analysis(contestant_df, output_lines)
    
    # Step 5: Run Module B - Partner Mixed Effects Analysis
    print("\n[STEP 5] Running Module B: Partner Analysis...")
    me_results, analysis_df = run_partner_analysis(contestant_df, history_df, output_lines)
    
    # Step 6: Compare Effects
    print("\n[STEP 6] Comparing Judge vs Fan Effects...")
    compare_judge_vs_fan_effects(ols_results, me_results, contestant_df, output_lines)
    
    # Step 7: Generate Key Findings
    print("\n[STEP 7] Generating Key Findings...")
    generate_key_findings(ols_results, me_results, contestant_df, output_lines)
    
    # Step 8: Save output
    output_file = '../data/task3_analysis_results.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n[DONE] Results saved to: {output_file}")
    print("=" * 60)
    
    # Also save contestant summary for reference
    contestant_df.to_csv('../data/task3_contestant_summary.csv', index=False)
    print(f"[DONE] Contestant summary saved to: ../data/task3_contestant_summary.csv")
    
    return contestant_df, ols_results, me_results


if __name__ == "__main__":
    contestant_df, ols_results, me_results = main()
