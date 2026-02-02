# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 V5 - 使用正确的反演数据

使用 fan_vote_results_final.csv 中已反演好的粉丝投票份额
每季使用固定 λ，计算综合得分 S = λ * j_share + (1-λ) * f_share
分歧度 = |综合排名 - 评委排名| 的加权和

Author: MCM 2026
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 数据加载
# ============================================================================

def load_data():
    """加载数据"""
    # 加载反演好的粉丝投票数据
    df_fan = pd.read_csv(r'd:\2026-repo\data\fan_vote_results_final.csv')
    
    # 加载完整数据（含人气分数）
    df_full = pd.read_csv(r'd:\2026-repo\data\task3_dataset_full.csv')
    
    # 填充0值
    for col in ['Celebrity_Average_Popularity_Score', 'ballroom_partner_Average_Popularity_Score']:
        for season in df_full['season'].unique():
            mask = df_full['season'] == season
            values = df_full.loc[mask, col].values
            non_zero = values[values > 0]
            if len(non_zero) > 0:
                df_full.loc[mask & (df_full[col] == 0), col] = non_zero.mean()
            else:
                df_full.loc[mask, col] = df_full[df_full[col] > 0][col].mean()
    
    # 合并人气数据
    pop_cols = ['season', 'week', 'celebrity_name', 'Celebrity_Average_Popularity_Score', 
                'ballroom_partner_Average_Popularity_Score', 'placement']
    df_fan = df_fan.merge(df_full[pop_cols].drop_duplicates(subset=['season', 'week', 'celebrity_name']),
                          on=['season', 'week', 'celebrity_name'], how='left')
    
    # 加载 Google Trends
    df_trends = pd.read_csv(r'd:\2026-repo\data\time_series_US_20040101-0800_20260202-0337.csv')
    df_trends.columns = ['Time', 'search_volume']
    df_trends['Time'] = pd.to_datetime(df_trends['Time'])
    
    schedule = {
        1: ('2005-06-01', '2005-07-06'), 2: ('2006-01-05', '2006-02-26'),
        3: ('2006-09-12', '2006-11-15'), 4: ('2007-03-19', '2007-05-22'),
        5: ('2007-09-24', '2007-11-27'), 6: ('2008-03-17', '2008-05-20'),
        7: ('2008-09-22', '2008-11-25'), 8: ('2009-03-09', '2009-05-19'),
        9: ('2009-09-21', '2009-11-24'), 10: ('2010-03-22', '2010-05-25'),
        11: ('2010-09-20', '2010-11-23'), 12: ('2011-03-21', '2011-05-24'),
        13: ('2011-09-19', '2011-11-22'), 14: ('2012-03-19', '2012-05-22'),
        15: ('2012-09-24', '2012-11-27'), 16: ('2013-03-18', '2013-05-21'),
        17: ('2013-09-16', '2013-11-26'), 18: ('2014-03-17', '2014-05-20'),
        19: ('2014-09-15', '2014-11-25'), 20: ('2015-03-16', '2015-05-19'),
        21: ('2015-09-14', '2015-11-24'), 22: ('2016-03-21', '2016-05-24'),
        23: ('2016-09-12', '2016-11-22'), 24: ('2017-03-20', '2017-05-23'),
        25: ('2017-09-18', '2017-11-21'), 26: ('2018-04-30', '2018-05-21'),
        27: ('2018-09-24', '2018-11-19'), 28: ('2019-09-16', '2019-11-25'),
        29: ('2020-09-14', '2020-11-23'), 30: ('2021-09-20', '2021-11-22'),
        31: ('2022-09-19', '2022-11-21'), 32: ('2023-09-26', '2023-11-21'),
        33: ('2024-09-17', '2024-11-26'), 34: ('2025-02-25', '2025-05-06'),
    }
    return df_full, df_fan, df_trends, schedule


# ============================================================================
# 提取历史淘汰
# ============================================================================

def extract_historical_elimination(df_season):
    """提取历史淘汰结果"""
    weeks = sorted(df_season['week'].unique())
    results = []
    
    for i, week in enumerate(weeks):
        week_df = df_season[df_season['week'] == week].copy()
        if len(week_df) <= 1:
            break
        
        # 找出本周被淘汰的人
        eliminated_rows = week_df[week_df['eliminated_this_week'] == True]
        eliminated_name = eliminated_rows['celebrity_name'].iloc[0] if len(eliminated_rows) > 0 else None
        
        results.append({
            'week': week, 'week_idx': i,
            'eliminated': eliminated_name,
            'week_data': week_df,
            'n': len(week_df)
        })
    
    return results


# ============================================================================
# 模拟淘汰（用于计算 Celebrity/Dancer/Satisfaction）
# ============================================================================

def simulate_elimination(df_season, lam, regime='percent'):
    """模拟淘汰过程，使用固定 λ"""
    weeks = sorted(df_season['week'].unique())
    all_contestants = df_season['celebrity_name'].unique().tolist()
    survival_weeks = {name: 0 for name in all_contestants}
    alive = set(all_contestants)
    
    for i, week in enumerate(weeks):
        week_df = df_season[(df_season['week'] == week) & (df_season['celebrity_name'].isin(alive))].copy()
        
        if len(week_df) <= 1:
            for name in alive:
                survival_weeks[name] = i + 1
            break
        
        # 计算综合得分 S = λ * j_share + (1-λ) * f_share
        j_sum = week_df['judge_total'].sum()
        f_sum = week_df['fan_vote_share'].sum()
        j_share = week_df['judge_total'] / j_sum if j_sum > 0 else 1 / len(week_df)
        f_share = week_df['fan_vote_share'] / f_sum if f_sum > 0 else 1 / len(week_df)
        
        week_df['score'] = lam * j_share + (1 - lam) * f_share
        
        # 淘汰得分最低者
        eliminated = week_df.loc[week_df['score'].idxmin(), 'celebrity_name']
        
        for name in alive:
            survival_weeks[name] = i + 1
        alive.remove(eliminated)
    
    for name in alive:
        survival_weeks[name] = len(weeks)
    
    return survival_weeks


# ============================================================================
# 计算分歧度 - 使用已反演的粉丝投票
# ============================================================================

def compute_discrepancy(historical_results, lam):
    """
    计算分歧度
    
    使用已反演的 fan_vote_share，计算：
    综合得分 S = λ * j_share + (1-λ) * f_share
    分歧度 = sum(|综合排名 - 评委排名|)
    
    λ 越小，粉丝权重越大，综合排名偏离评委越远，分歧度越大
    λ 越大，评委权重越大，综合排名接近评委，分歧度越小
    """
    total = 0
    
    for result in historical_results:
        week_data = result['week_data']
        n = result['n']
        
        if n <= 1:
            continue
        
        J = week_data['judge_total'].values.astype(float)
        F = week_data['fan_vote_share'].values.astype(float)
        
        j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
        f_share = F / F.sum() if F.sum() > 0 else np.ones(n) / n
        
        # 综合得分
        S = lam * j_share + (1 - lam) * f_share
        
        # 排名（1=最高分）
        judge_ranks = rankdata(-j_share, method='average')
        combined_ranks = rankdata(-S, method='average')
        
        # 分歧度 = 排名差异之和
        week_disc = np.sum(np.abs(combined_ranks - judge_ranks))
        total += week_disc
    
    return total


# ============================================================================
# 计算其他指标
# ============================================================================

def compute_celebrity_benefit(df_season, survival_weeks):
    total = 0
    for name, weeks in survival_weeks.items():
        data = df_season[df_season['celebrity_name'] == name]
        if len(data) > 0:
            pop = data['Celebrity_Average_Popularity_Score'].iloc[0]
            total += (pop if pd.notna(pop) else 0) * weeks
    return total


def compute_dancer_effect(df_season, survival_weeks):
    total = 0
    for name, weeks in survival_weeks.items():
        data = df_season[df_season['celebrity_name'] == name]
        if len(data) > 0:
            pop = data['ballroom_partner_Average_Popularity_Score'].iloc[0]
            total += (pop if pd.notna(pop) else 0) * weeks
    return total


def compute_ccvi(df_trends, schedule, season):
    if season not in schedule:
        return 0
    start, end = schedule[season]
    start_dt = pd.to_datetime(start) - pd.DateOffset(weeks=1)
    end_dt = pd.to_datetime(end) + pd.DateOffset(weeks=1)
    mask = (df_trends['Time'] >= start_dt) & (df_trends['Time'] <= end_dt)
    data = df_trends[mask]['search_volume'].values
    if len(data) == 0:
        return 0
    return 0.3 * np.max(data) + 0.5 * np.sum(data) + 0.2 * max(0, np.max(np.diff(data)) if len(data) > 1 else 0)


def compute_satisfaction(df_season, survival_weeks):
    stats = []
    for name, weeks in survival_weeks.items():
        data = df_season[df_season['celebrity_name'] == name]
        if len(data) == 0:
            continue
        pcts = []
        for w in data['week'].unique():
            w_df = df_season[df_season['week'] == w]
            n = len(w_df)
            if n == 0:
                continue
            fvs = data[data['week'] == w]['fan_vote_share'].iloc[0]
            rank = np.sum(w_df['fan_vote_share'].values > fvs) + 1
            pcts.append(rank / n)
        stats.append({'mean_pct': np.mean(pcts) if pcts else 0.5, 'weeks': weeks})
    
    if len(stats) < 3:
        return 0
    df = pd.DataFrame(stats)
    df['placement'] = df['weeks'].rank(ascending=False) / len(df)
    corr, _ = spearmanr(df['mean_pct'], df['placement'])
    return corr if not np.isnan(corr) else 0


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("Task 4: 动态权重优化系统 V5 - 使用正确的反演数据")
    print("每季使用固定 λ，搜索范围 [0.2, 0.8]，步长 0.05")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    df_full, df_fan, df_trends, schedule = load_data()
    
    weights = (0.2, 0.1, 0.3, 0.2, 0.2)
    lambda_options = [round(x * 0.05, 2) for x in range(4, 17)]  # 0.20, 0.25, ..., 0.80
    print(f"    λ候选值: {lambda_options}")
    print(f"\n企业权重: w1=0.2, w2=0.1, w3=0.3, w4=0.2, w5=0.2")
    print("收益公式: w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy")
    print("\n分歧度计算: S = λ*j_share + (1-λ)*f_share, disc = Σ|combined_rank - judge_rank|")
    
    # 预热：计算归一化参数
    print("\n[2/3] 预热阶段...")
    all_metrics = {'celebrity': [], 'dancer': [], 'ccvi': [], 'satisfaction': [], 'discrepancy': []}
    
    for season in sorted(df_fan['season'].unique()):
        df_s = df_fan[df_fan['season'] == season].copy()
        df_s['Celebrity_Average_Popularity_Score'] = df_s['Celebrity_Average_Popularity_Score'].fillna(0)
        df_s['ballroom_partner_Average_Popularity_Score'] = df_s['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        survival = simulate_elimination(df_s, 0.5)
        historical = extract_historical_elimination(df_s)
        
        all_metrics['celebrity'].append(compute_celebrity_benefit(df_s, survival))
        all_metrics['dancer'].append(compute_dancer_effect(df_s, survival))
        all_metrics['ccvi'].append(compute_ccvi(df_trends, schedule, season))
        all_metrics['satisfaction'].append(compute_satisfaction(df_s, survival))
        all_metrics['discrepancy'].append(compute_discrepancy(historical, 0.5))
    
    scale_params = {}
    for col in all_metrics:
        vals = np.array(all_metrics[col])
        mean, std = np.mean(vals), np.std(vals)
        z = (vals - mean) / std if std > 0 else np.zeros_like(vals)
        scale_params[col] = {'mean': mean, 'std': std, 'z_min': np.min(z), 'z_max': np.max(z)}
    
    def normalize(val, col):
        p = scale_params[col]
        if p['std'] == 0:
            return 0.5
        z = (val - p['mean']) / p['std']
        if p['z_max'] == p['z_min']:
            return 0.5
        return np.clip((z - p['z_min']) / (p['z_max'] - p['z_min']), 0, 1)
    
    # 搜索最佳 λ
    print("\n[3/3] 搜索每季最佳 λ...")
    print("=" * 70)
    print(f"{'Season':<8} {'Regime':<10} {'Weeks':<6} {'Best λ':<8} {'Benefit':<10} {'Baseline':<10} {'Improve':<10}")
    print("-" * 70)
    
    results = []
    
    for season in sorted(df_fan['season'].unique()):
        df_s = df_fan[df_fan['season'] == season].copy()
        df_s['Celebrity_Average_Popularity_Score'] = df_s['Celebrity_Average_Popularity_Score'].fillna(0)
        df_s['ballroom_partner_Average_Popularity_Score'] = df_s['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        n_weeks = df_s['week'].nunique()
        regime = df_s['regime'].iloc[0] if 'regime' in df_s.columns else 'percent'
        historical = extract_historical_elimination(df_s)
        ccvi = compute_ccvi(df_trends, schedule, season)
        
        best_lam = 0.5
        best_benefit = -np.inf
        
        for lam in lambda_options:
            survival = simulate_elimination(df_s, lam)
            
            celebrity = compute_celebrity_benefit(df_s, survival)
            dancer = compute_dancer_effect(df_s, survival)
            satisfaction = compute_satisfaction(df_s, survival)
            discrepancy = compute_discrepancy(historical, lam)
            
            w1, w2, w3, w4, w5 = weights
            c_norm = normalize(celebrity, 'celebrity')
            d_norm = normalize(dancer, 'dancer')
            ccvi_norm = normalize(ccvi, 'ccvi')
            s_norm = normalize(satisfaction, 'satisfaction')
            disc_norm = normalize(discrepancy, 'discrepancy')
            
            # Discrepancy 取负值（分歧度越小越好）
            benefit = w1 * c_norm + w2 * d_norm + w3 * ccvi_norm + w4 * s_norm - w5 * disc_norm
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_lam = lam
        
        # 计算基准 (λ=0.5)
        survival_base = simulate_elimination(df_s, 0.5)
        celebrity_base = compute_celebrity_benefit(df_s, survival_base)
        dancer_base = compute_dancer_effect(df_s, survival_base)
        satisfaction_base = compute_satisfaction(df_s, survival_base)
        discrepancy_base = compute_discrepancy(historical, 0.5)
        
        baseline = (w1 * normalize(celebrity_base, 'celebrity') + 
                   w2 * normalize(dancer_base, 'dancer') + 
                   w3 * normalize(ccvi, 'ccvi') + 
                   w4 * normalize(satisfaction_base, 'satisfaction') - 
                   w5 * normalize(discrepancy_base, 'discrepancy'))
        
        improvement = best_benefit - baseline
        
        print(f"{season:<8} {regime:<10} {n_weeks:<6} {best_lam:<8.2f} {best_benefit:<10.4f} {baseline:<10.4f} {improvement:+.4f}")
        
        results.append({
            'season': season, 'regime': regime, 'n_weeks': n_weeks,
            'optimal_lambda': best_lam,
            'benefit': best_benefit,
            'baseline': baseline,
            'improvement': improvement
        })
    
    # 保存
    print("\n" + "=" * 70)
    results_df = pd.DataFrame(results)
    results_df.to_csv(r'd:\2026-repo\data\task4_optimal_results_v5.csv', index=False)
    
    print("\n优化结果汇总")
    print("=" * 70)
    print(f"\n最优λ分布:")
    print(results_df['optimal_lambda'].value_counts().sort_index())
    
    print(f"\n平均最优λ: {results_df['optimal_lambda'].mean():.3f}")
    print(f"平均收益: {results_df['benefit'].mean():.4f}")
    print(f"平均改进: {results_df['improvement'].mean():.4f}")
    print(f"改进 > 0 的赛季数: {(results_df['improvement'] > 0).sum()} / {len(results_df)}")
    
    print("\n按赛制分组:")
    print(results_df.groupby('regime')[['optimal_lambda', 'benefit', 'improvement']].mean().round(4))
    
    print(f"\n完成！结果保存至 data/task4_optimal_results_v5.csv")


if __name__ == "__main__":
    main()
