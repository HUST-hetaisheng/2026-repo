# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 - 正确版

正确逻辑：
1. 反演使用固定规则 S = j_share + f_share（论文模型，已完成）
2. Task 4 是提出新规则：S_new = λ * j_share + (1-λ) * f_share
3. 用已反演的 fan_vote_share 评估不同 λ 下的淘汰效果
4. 分歧度 = |fan_rank - judge_rank| （已反演的 f 和 j 之间的差异，与 λ 无关）

λ 的作用：
- 决定新规则下谁会被淘汰
- 影响 Celebrity Benefit, Dancer Effect, Satisfaction（因为存活周数变了）
- 分歧度是固定的（基于已反演的数据）

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
    # 加载已反演好的粉丝投票数据（使用固定规则 S = j + f）
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
    return df_fan, df_full, df_trends, schedule


# ============================================================================
# 模拟淘汰（使用新规则 S = λ*j + (1-λ)*f）
# ============================================================================

def simulate_elimination_with_new_rule(df_season, lam):
    """
    使用新规则模拟淘汰过程
    新规则: S = λ * j_share + (1-λ) * f_share
    每周淘汰综合得分最低的人
    """
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
        
        # 计算新规则的综合得分
        j_sum = week_df['judge_total'].sum()
        f_sum = week_df['fan_vote_share'].sum()
        j_share = week_df['judge_total'] / j_sum if j_sum > 0 else 1 / len(week_df)
        f_share = week_df['fan_vote_share'] / f_sum if f_sum > 0 else 1 / len(week_df)
        
        # 新规则: S = λ * j_share + (1-λ) * f_share
        week_df['score'] = lam * j_share.values + (1 - lam) * f_share.values
        
        # 淘汰得分最低者
        eliminated = week_df.loc[week_df['score'].idxmin(), 'celebrity_name']
        
        for name in alive:
            survival_weeks[name] = i + 1
        alive.remove(eliminated)
    
    for name in alive:
        survival_weeks[name] = len(weeks)
    
    return survival_weeks


# ============================================================================
# 计算分歧度（基于已反演的数据，与 λ 无关）
# ============================================================================

def compute_discrepancy(df_season):
    """
    计算分歧度 = Σ|fan_rank - judge_rank| * n
    
    这是已反演的 f 和 j 之间的固有差异，与新规则的 λ 无关
    """
    total = 0
    
    for week in df_season['week'].unique():
        week_df = df_season[df_season['week'] == week]
        n = len(week_df)
        if n <= 1:
            continue
        
        J = week_df['judge_total'].values
        F = week_df['fan_vote_share'].values
        
        j_ranks = rankdata(-J, method='average')
        f_ranks = rankdata(-F, method='average')
        
        week_disc = np.sum(np.abs(f_ranks - j_ranks))
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
    """计算观众满意度：粉丝排名 vs 最终名次的相关性"""
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
    print("Task 4: 动态权重优化系统 - 正确版")
    print("使用已反演的 fan_vote_share，评估新规则 S = λ*j + (1-λ)*f")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    df_fan, df_full, df_trends, schedule = load_data()
    
    weights = (0.2, 0.1, 0.3, 0.2, 0.2)
    lambda_options = [round(x * 0.1, 1) for x in range(2, 9)]  # 0.2, 0.3, ..., 0.8
    print(f"    λ候选值: {lambda_options}")
    print(f"\n企业权重: w1=0.2, w2=0.1, w3=0.3, w4=0.2, w5=0.2")
    print("收益公式: w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy")
    print("\n注意：分歧度基于已反演数据，与新规则的 λ 无关")
    
    # 预热：计算归一化参数
    print("\n[2/3] 预热阶段（计算归一化参数）...")
    all_metrics = {'celebrity': [], 'dancer': [], 'ccvi': [], 'satisfaction': [], 'discrepancy': []}
    
    for season in sorted(df_fan['season'].unique()):
        df_s = df_fan[df_fan['season'] == season].copy()
        df_s['Celebrity_Average_Popularity_Score'] = df_s['Celebrity_Average_Popularity_Score'].fillna(0)
        df_s['ballroom_partner_Average_Popularity_Score'] = df_s['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        survival = simulate_elimination_with_new_rule(df_s, 0.5)
        
        all_metrics['celebrity'].append(compute_celebrity_benefit(df_s, survival))
        all_metrics['dancer'].append(compute_dancer_effect(df_s, survival))
        all_metrics['ccvi'].append(compute_ccvi(df_trends, schedule, season))
        all_metrics['satisfaction'].append(compute_satisfaction(df_s, survival))
        all_metrics['discrepancy'].append(compute_discrepancy(df_s))
    
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
        ccvi = compute_ccvi(df_trends, schedule, season)
        discrepancy = compute_discrepancy(df_s)  # 固定，与 λ 无关
        
        best_lam = 0.5
        best_benefit = -np.inf
        
        for lam in lambda_options:
            survival = simulate_elimination_with_new_rule(df_s, lam)
            
            celebrity = compute_celebrity_benefit(df_s, survival)
            dancer = compute_dancer_effect(df_s, survival)
            satisfaction = compute_satisfaction(df_s, survival)
            
            w1, w2, w3, w4, w5 = weights
            c_norm = normalize(celebrity, 'celebrity')
            d_norm = normalize(dancer, 'dancer')
            ccvi_norm = normalize(ccvi, 'ccvi')
            s_norm = normalize(satisfaction, 'satisfaction')
            disc_norm = normalize(discrepancy, 'discrepancy')
            
            # Discrepancy 取负值
            benefit = w1 * c_norm + w2 * d_norm + w3 * ccvi_norm + w4 * s_norm - w5 * disc_norm
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_lam = lam
        
        # 基准 (λ=0.5)
        survival_base = simulate_elimination_with_new_rule(df_s, 0.5)
        celebrity_base = compute_celebrity_benefit(df_s, survival_base)
        dancer_base = compute_dancer_effect(df_s, survival_base)
        satisfaction_base = compute_satisfaction(df_s, survival_base)
        
        baseline = (w1 * normalize(celebrity_base, 'celebrity') + 
                   w2 * normalize(dancer_base, 'dancer') + 
                   w3 * normalize(ccvi, 'ccvi') + 
                   w4 * normalize(satisfaction_base, 'satisfaction') - 
                   w5 * normalize(discrepancy, 'discrepancy'))
        
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
    results_df.to_csv(r'd:\2026-repo\data\task4_optimal_results_correct.csv', index=False)
    
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
    
    print(f"\n完成！结果保存至 data/task4_optimal_results_correct.csv")


if __name__ == "__main__":
    main()
