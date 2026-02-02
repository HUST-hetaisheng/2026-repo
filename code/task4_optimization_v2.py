# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 V2

逻辑：
1. 预热阶段：用λ=0.5对所有赛季计算5个指标，确定放缩系数
2. 搜索阶段：对每个赛季，网格搜索每周最佳λ向量

Author: MCM 2026
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 数据加载
# ============================================================================

def load_data():
    """加载所有需要的数据"""
    # 主数据集
    df_main = pd.read_csv(r'd:\2026-repo\data\task3_dataset_full_zscored.csv')
    
    # Google Trends 时间序列
    df_trends = pd.read_csv(r'd:\2026-repo\data\time_series_US_20040101-0800_20260202-0337.csv')
    df_trends.columns = ['Time', 'search_volume']
    df_trends['Time'] = pd.to_datetime(df_trends['Time'])
    
    # 赛程表
    schedule = {
        1: ('2005-06-01', '2005-07-06'),
        2: ('2006-01-05', '2006-02-26'),
        3: ('2006-09-12', '2006-11-15'),
        4: ('2007-03-19', '2007-05-22'),
        5: ('2007-09-24', '2007-11-27'),
        6: ('2008-03-17', '2008-05-20'),
        7: ('2008-09-22', '2008-11-25'),
        8: ('2009-03-09', '2009-05-19'),
        9: ('2009-09-21', '2009-11-24'),
        10: ('2010-03-22', '2010-05-25'),
        11: ('2010-09-20', '2010-11-23'),
        12: ('2011-03-21', '2011-05-24'),
        13: ('2011-09-19', '2011-11-22'),
        14: ('2012-03-19', '2012-05-22'),
        15: ('2012-09-24', '2012-11-27'),
        16: ('2013-03-18', '2013-05-21'),
        17: ('2013-09-16', '2013-11-26'),
        18: ('2014-03-17', '2014-05-20'),
        19: ('2014-09-15', '2014-11-25'),
        20: ('2015-03-16', '2015-05-19'),
        21: ('2015-09-14', '2015-11-24'),
        22: ('2016-03-21', '2016-05-24'),
        23: ('2016-09-12', '2016-11-22'),
        24: ('2017-03-20', '2017-05-23'),
        25: ('2017-09-18', '2017-11-21'),
        26: ('2018-04-30', '2018-05-21'),
        27: ('2018-09-24', '2018-11-19'),
        28: ('2019-09-16', '2019-11-25'),
        29: ('2020-09-14', '2020-11-23'),
        30: ('2021-09-20', '2021-11-22'),
        31: ('2022-09-19', '2022-11-21'),
        32: ('2023-09-26', '2023-12-05'),
        33: ('2024-09-17', '2024-11-26'),
        34: ('2025-09-16', '2025-11-25'),
    }
    
    return df_main, df_trends, schedule


# ============================================================================
# 模拟淘汰过程
# ============================================================================

def simulate_elimination(df_season, lambda_vec):
    """
    给定一个赛季的数据和每周评委权重λ向量，模拟淘汰过程
    
    综合得分 = λ * judge_score_pct + (1-λ) * fan_vote_pct
    每周淘汰综合得分最低的选手
    
    返回：每个选手的模拟存活周数
    """
    weeks = sorted(df_season['week'].unique())
    n_weeks = len(weeks)
    
    # 获取所有选手
    all_contestants = df_season['celebrity_name'].unique().tolist()
    
    # 记录每个选手的存活周数
    survival_weeks = {name: 0 for name in all_contestants}
    
    # 当前存活选手
    alive = set(all_contestants)
    
    for i, week in enumerate(weeks):
        lam = lambda_vec[i] if i < len(lambda_vec) else 0.5
        
        # 该周数据（只看存活选手）
        week_df = df_season[(df_season['week'] == week) & 
                            (df_season['celebrity_name'].isin(alive))].copy()
        
        if len(week_df) <= 1:
            # 最后一人或没人，结束
            for name in alive:
                survival_weeks[name] = i + 1
            break
        
        # 计算排名百分比
        # 粉丝排名（fan_vote_share 越高越好）
        week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False)
        week_df['fan_rank_pct'] = week_df['fan_rank'] / len(week_df)
        
        # 评委排名（judge_total 越高越好）
        week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False)
        week_df['judge_rank_pct'] = week_df['judge_rank'] / len(week_df)
        
        # 综合得分（排名百分比越低越好）
        week_df['combined_rank_pct'] = lam * week_df['judge_rank_pct'] + (1 - lam) * week_df['fan_rank_pct']
        
        # 更新存活周数
        for name in alive:
            survival_weeks[name] = i + 1
        
        # 淘汰综合排名最差（百分比最高）的选手
        eliminated = week_df.loc[week_df['combined_rank_pct'].idxmax(), 'celebrity_name']
        alive.remove(eliminated)
    
    # 剩余选手存活到最后
    for name in alive:
        survival_weeks[name] = n_weeks
    
    return survival_weeks


# ============================================================================
# 计算五个指标（基于模拟的存活周数）
# ============================================================================

def compute_celebrity_benefit(df_season, survival_weeks):
    """
    名人收益 = Σ (social_media_popularity + google_search_volume) / 2 * survived_weeks
    """
    total = 0
    for name, weeks in survival_weeks.items():
        celeb_data = df_season[df_season['celebrity_name'] == name]
        if len(celeb_data) == 0:
            continue
        # 取最后一行的人气数据
        social = celeb_data['social_media_popularity'].iloc[-1]
        google = celeb_data['google_search_volume'].iloc[-1]
        # 热度（这里用原始zscore值，最后统一放缩）
        heat = (social + google) / 2
        total += heat * weeks
    return total


def compute_dancer_effect(df_season, survival_weeks):
    """
    舞者效应 = Σ partner_popularity * survived_weeks
    """
    total = 0
    for name, weeks in survival_weeks.items():
        celeb_data = df_season[df_season['celebrity_name'] == name]
        if len(celeb_data) == 0:
            continue
        # 取舞伴热度
        partner_pop = celeb_data['ballroom_partner_Average_Popularity_Score'].iloc[0]
        total += partner_pop * weeks
    return total


def compute_ccvi(df_trends, schedule, season):
    """
    CCVI = 0.3*P + 0.5*I + 0.2*D（归一化到0-100）
    注意：CCVI只与赛季相关，不受λ影响
    """
    if season not in schedule:
        return 0
    
    start, end = schedule[season]
    start_dt = pd.to_datetime(start) - pd.DateOffset(weeks=1)
    end_dt = pd.to_datetime(end) + pd.DateOffset(weeks=1)
    
    mask = (df_trends['Time'] >= start_dt) & (df_trends['Time'] <= end_dt)
    season_data = df_trends[mask]['search_volume'].values
    
    if len(season_data) == 0:
        return 0
    
    P = np.max(season_data)
    I = np.sum(season_data)
    D = np.max(np.diff(season_data)) if len(season_data) > 1 else 0
    
    # 简化：直接返回原始值，后面统一放缩
    return 0.3 * P + 0.5 * I + 0.2 * max(0, D)


def compute_fan_satisfaction(df_season, survival_weeks):
    """
    观众满意度 = Spearman(平均粉丝排名, 最终名次)
    最终名次由存活周数决定
    """
    stats = []
    
    for name, weeks in survival_weeks.items():
        celeb_data = df_season[df_season['celebrity_name'] == name]
        if len(celeb_data) == 0:
            continue
        
        # 计算平均粉丝排名百分比
        fan_rank_pcts = []
        for week in celeb_data['week'].unique():
            week_all = df_season[df_season['week'] == week]
            n = len(week_all)
            celeb_week = week_all[week_all['celebrity_name'] == name]
            if len(celeb_week) > 0:
                # 在该周所有人中排名
                week_all_sorted = week_all.sort_values('fan_vote_share', ascending=False)
                rank_in_week = week_all_sorted['celebrity_name'].tolist().index(name) + 1
                fan_rank_pcts.append(rank_in_week / n)
        
        mean_fan_rank_pct = np.mean(fan_rank_pcts) if fan_rank_pcts else 0.5
        
        stats.append({
            'name': name,
            'mean_fan_rank_pct': mean_fan_rank_pct,
            'survived_weeks': weeks
        })
    
    if len(stats) < 3:
        return 0
    
    stats_df = pd.DataFrame(stats)
    # 存活周数越多，名次越好（名次数值越小）
    stats_df['placement'] = stats_df['survived_weeks'].rank(ascending=False)
    stats_df['placement_pct'] = stats_df['placement'] / len(stats_df)
    
    corr, _ = spearmanr(stats_df['mean_fan_rank_pct'], stats_df['placement_pct'])
    return corr if not np.isnan(corr) else 0


def compute_discrepancy(df_season, lambda_vec):
    """
    分歧度 = Σ_week Σ_contestant |fan_rank_pct - judge_rank_pct|
    这个不受模拟影响，直接用原始数据计算
    """
    weeks = sorted(df_season['week'].unique())
    total_discrepancy = 0
    
    for week in weeks:
        week_df = df_season[df_season['week'] == week].copy()
        n = len(week_df)
        if n == 0:
            continue
        
        week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False)
        week_df['fan_rank_pct'] = week_df['fan_rank'] / n
        
        week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False)
        week_df['judge_rank_pct'] = week_df['judge_rank'] / n
        
        week_df['discrepancy'] = np.abs(week_df['fan_rank_pct'] - week_df['judge_rank_pct'])
        total_discrepancy += week_df['discrepancy'].sum()
    
    return total_discrepancy


# ============================================================================
# 计算总效益（单个赛季）
# ============================================================================

def compute_season_benefit(df_season, df_trends, schedule, season, lambda_vec, 
                           scale_params, weights):
    """
    计算给定λ向量下的总效益
    
    scale_params: 五个指标的 (min, max) 元组列表，用于Min-Max归一化
    weights: 五个权重 [w1, w2, w3, w4, w5]
    """
    w1, w2, w3, w4, w5 = weights
    
    # 模拟淘汰
    survival_weeks = simulate_elimination(df_season, lambda_vec)
    
    # 计算五个指标
    celebrity = compute_celebrity_benefit(df_season, survival_weeks)
    dancer = compute_dancer_effect(df_season, survival_weeks)
    ccvi = compute_ccvi(df_trends, schedule, season)
    satisfaction = compute_fan_satisfaction(df_season, survival_weeks)
    discrepancy = compute_discrepancy(df_season, lambda_vec)
    
    # Min-Max归一化到[0,1]
    def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    celeb_norm = normalize(celebrity, scale_params['celebrity']['min'], scale_params['celebrity']['max'])
    dancer_norm = normalize(dancer, scale_params['dancer']['min'], scale_params['dancer']['max'])
    ccvi_norm = normalize(ccvi, scale_params['ccvi']['min'], scale_params['ccvi']['max'])
    satisfaction_norm = normalize(satisfaction, scale_params['satisfaction']['min'], scale_params['satisfaction']['max'])
    # 分歧度需要反转：原值越小越好，归一化后越大越好
    discrepancy_norm = 1 - normalize(discrepancy, scale_params['discrepancy']['min'], scale_params['discrepancy']['max'])
    
    # 加权求和（所有项都在[0,1]范围内）
    total_benefit = (
        w1 * celeb_norm +
        w2 * dancer_norm +
        w3 * ccvi_norm +
        w4 * satisfaction_norm +
        w5 * discrepancy_norm  # 已反转，越大越好
    )
    
    return total_benefit, {
        'celebrity': celebrity,
        'dancer': dancer,
        'ccvi': ccvi,
        'satisfaction': satisfaction,
        'discrepancy': discrepancy
    }


# ============================================================================
# 预热阶段：用λ=0.5计算Min-Max归一化参数
# ============================================================================

def warmup_phase(df_main, df_trends, schedule):
    """
    用λ=0.5对所有赛季计算5个指标，确定Min-Max归一化的min和max参数
    """
    print("=" * 60)
    print("预热阶段：用λ=0.5计算各赛季指标")
    print("=" * 60)
    
    all_metrics = []
    
    for season in sorted(df_main['season'].unique()):
        df_season = df_main[df_main['season'] == season]
        n_weeks = df_season['week'].nunique()
        
        # λ=0.5 的向量
        lambda_vec = [0.5] * n_weeks
        
        # 模拟并计算指标
        survival_weeks = simulate_elimination(df_season, lambda_vec)
        
        celebrity = compute_celebrity_benefit(df_season, survival_weeks)
        dancer = compute_dancer_effect(df_season, survival_weeks)
        ccvi = compute_ccvi(df_trends, schedule, season)
        satisfaction = compute_fan_satisfaction(df_season, survival_weeks)
        discrepancy = compute_discrepancy(df_season, lambda_vec)
        
        all_metrics.append({
            'season': season,
            'celebrity': celebrity,
            'dancer': dancer,
            'ccvi': ccvi,
            'satisfaction': satisfaction,
            'discrepancy': discrepancy
        })
    
    metrics_df = pd.DataFrame(all_metrics)
    
    print("\n各指标原始值统计：")
    print(metrics_df.describe().round(2))
    
    # 计算Min-Max归一化参数（每个指标的min和max）
    scale_params = {
        'celebrity': {'min': metrics_df['celebrity'].min(), 'max': metrics_df['celebrity'].max()},
        'dancer': {'min': metrics_df['dancer'].min(), 'max': metrics_df['dancer'].max()},
        'ccvi': {'min': metrics_df['ccvi'].min(), 'max': metrics_df['ccvi'].max()},
        'satisfaction': {'min': metrics_df['satisfaction'].min(), 'max': metrics_df['satisfaction'].max()},
        'discrepancy': {'min': metrics_df['discrepancy'].min(), 'max': metrics_df['discrepancy'].max()},
    }
    
    print("\nMin-Max归一化参数：")
    for col, params in scale_params.items():
        print(f"  {col}: min={params['min']:.2f}, max={params['max']:.2f}")
    
    # 验证归一化后的范围（应该都是[0,1]）
    print("\n归一化后各指标范围（验证）：")
    for col, params in scale_params.items():
        if params['max'] != params['min']:
            scaled = (metrics_df[col] - params['min']) / (params['max'] - params['min'])
            print(f"  {col}: [{scaled.min():.3f}, {scaled.max():.3f}]")
        else:
            print(f"  {col}: [0.500, 0.500] (常数)")
    
    return metrics_df, scale_params


# ============================================================================
# 搜索阶段：网格搜索最佳λ向量
# ============================================================================

def search_optimal_lambda(df_season, df_trends, schedule, season, 
                          scale_params, weights, lambda_options=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    网格搜索最佳λ向量
    
    为减少搜索空间，假设所有周使用相同的λ值
    """
    n_weeks = df_season['week'].nunique()
    
    best_lambda = None
    best_benefit = -np.inf
    best_metrics = None
    
    # 方案1：所有周使用相同λ
    for lam in lambda_options:
        lambda_vec = [lam] * n_weeks
        benefit, metrics = compute_season_benefit(
            df_season, df_trends, schedule, season, 
            lambda_vec, scale_params, weights
        )
        
        if benefit > best_benefit:
            best_benefit = benefit
            best_lambda = lambda_vec
            best_metrics = metrics
    
    return best_lambda, best_benefit, best_metrics


def search_optimal_lambda_full(df_season, df_trends, schedule, season, 
                               scale_params, weights, lambda_options=[0.2, 0.5, 0.8]):
    """
    完整网格搜索：每周独立选择λ
    注意：搜索空间 = len(lambda_options) ^ n_weeks，可能很大
    """
    n_weeks = df_season['week'].nunique()
    
    # 限制搜索，如果周数太多则使用简化版
    if n_weeks > 6:
        return search_optimal_lambda(df_season, df_trends, schedule, season, 
                                     scale_params, weights, lambda_options)
    
    best_lambda = None
    best_benefit = -np.inf
    best_metrics = None
    
    # 生成所有可能的λ组合
    for lambda_vec in product(lambda_options, repeat=n_weeks):
        lambda_vec = list(lambda_vec)
        benefit, metrics = compute_season_benefit(
            df_season, df_trends, schedule, season, 
            lambda_vec, scale_params, weights
        )
        
        if benefit > best_benefit:
            best_benefit = benefit
            best_lambda = lambda_vec
            best_metrics = metrics
    
    return best_lambda, best_benefit, best_metrics


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("Task 4: 动态权重优化系统 V2")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    df_main, df_trends, schedule = load_data()
    print(f"    主数据集: {len(df_main)} 条记录")
    print(f"    覆盖赛季: {df_main['season'].min()} - {df_main['season'].max()}")
    
    # 设定企业权重（可调整）
    weights = [0.2, 0.1, 0.3, 0.2, 0.2]  # w1-w5: celebrity, dancer, ccvi, satisfaction, discrepancy
    print(f"\n企业权重设定: w1={weights[0]}, w2={weights[1]}, w3={weights[2]}, w4={weights[3]}, w5={weights[4]}")
    
    # 预热阶段
    print("\n[2/4] 预热阶段...")
    warmup_metrics, scale_params = warmup_phase(df_main, df_trends, schedule)
    
    # 保存预热结果
    warmup_metrics.to_csv(r'd:\2026-repo\data\task4_warmup_metrics.csv', index=False)
    
    # 搜索阶段
    print("\n" + "=" * 60)
    print("[3/4] 搜索阶段：寻找各赛季最优λ")
    print("=" * 60)
    
    lambda_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    for season in sorted(df_main['season'].unique()):
        df_season = df_main[df_main['season'] == season]
        n_weeks = df_season['week'].nunique()
        
        # 搜索最优λ
        best_lambda, best_benefit, best_metrics = search_optimal_lambda(
            df_season, df_trends, schedule, season,
            scale_params, weights, lambda_options
        )
        
        # 也计算λ=0.5时的效益作为基准
        baseline_lambda = [0.5] * n_weeks
        baseline_benefit, baseline_metrics = compute_season_benefit(
            df_season, df_trends, schedule, season,
            baseline_lambda, scale_params, weights
        )
        
        improvement = (best_benefit - baseline_benefit) / abs(baseline_benefit) * 100 if baseline_benefit != 0 else 0
        
        results.append({
            'season': season,
            'n_weeks': n_weeks,
            'optimal_lambda': best_lambda[0],  # 简化版：统一λ
            'baseline_benefit': baseline_benefit,
            'optimal_benefit': best_benefit,
            'improvement_pct': improvement,
            **{f'opt_{k}': v for k, v in best_metrics.items()}
        })
        
        print(f"  Season {season:2d}: λ*={best_lambda[0]:.1f}, "
              f"Benefit: {baseline_benefit:.3f} → {best_benefit:.3f} "
              f"({improvement:+.1f}%)")
    
    results_df = pd.DataFrame(results)
    
    # 保存结果
    output_path = r'd:\2026-repo\data\task4_optimal_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")
    
    # 详细分析第27季（Bobby Bones案例）
    print("\n" + "=" * 60)
    print("[4/4] 详细分析：第27季 (Bobby Bones)")
    print("=" * 60)
    
    df_s27 = df_main[df_main['season'] == 27]
    n_weeks = df_s27['week'].nunique()
    
    print(f"第27季: {n_weeks} 周")
    
    # 完整搜索（每周独立λ）
    print("\n完整网格搜索 (λ ∈ {0.2, 0.5, 0.8})...")
    best_lambda_full, best_benefit_full, best_metrics_full = search_optimal_lambda_full(
        df_s27, df_trends, schedule, 27,
        scale_params, weights, [0.2, 0.5, 0.8]
    )
    
    print(f"\n最优λ向量 (每周):")
    for i, lam in enumerate(best_lambda_full):
        print(f"    Week {i+1}: λ = {lam}")
    
    print(f"\n最优总效益: {best_benefit_full:.4f}")
    print(f"各分项指标:")
    for k, v in best_metrics_full.items():
        print(f"    {k}: {v:.4f}")
    
    # 保存第27季详细结果
    lambda_df = pd.DataFrame({
        'week': range(1, len(best_lambda_full) + 1),
        'optimal_lambda': best_lambda_full
    })
    lambda_df.to_csv(r'd:\2026-repo\data\task4_s27_optimal_lambda.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Task 4 完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
