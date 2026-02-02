# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 V5 (快速版)

只跑5个代表性赛季：3个percent + 2个bottom2
每周 λ ∈ {0.2, 0.4, 0.6, 0.8}，4种选择
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
from itertools import product
import time

# ============================================================================
# 数据加载
# ============================================================================

def load_data():
    df_full = pd.read_csv(r'd:\2026-repo\data\task3_dataset_full.csv')
    
    # 填充0值
    for col in ['Celebrity_Average_Popularity_Score', 'ballroom_partner_Average_Popularity_Score']:
        global_mean = df_full[df_full[col] > 0][col].mean()
        df_full[col] = df_full[col].replace(0, global_mean)
    
    df_fan = pd.read_csv(r'd:\2026-repo\data\fan_vote_results_final.csv')
    pop_cols = ['season', 'week', 'celebrity_name', 'Celebrity_Average_Popularity_Score', 
                'ballroom_partner_Average_Popularity_Score', 'placement']
    df_fan = df_fan.merge(df_full[pop_cols].drop_duplicates(subset=['season', 'week', 'celebrity_name']),
                          on=['season', 'week', 'celebrity_name'], how='left')
    
    df_trends = pd.read_csv(r'd:\2026-repo\data\time_series_US_20040101-0800_20260202-0337.csv')
    df_trends.columns = ['Time', 'search_volume']
    df_trends['Time'] = pd.to_datetime(df_trends['Time'])
    
    schedule = {
        3: ('2006-09-12', '2006-11-15'), 10: ('2010-03-22', '2010-05-25'),
        20: ('2015-03-16', '2015-05-19'), 28: ('2019-09-16', '2019-11-25'),
        32: ('2023-09-26', '2023-11-21'),
    }
    return df_full, df_fan, df_trends, schedule


# ============================================================================
# 提取历史淘汰
# ============================================================================

def extract_historical_elimination(df_season):
    weeks = sorted(df_season['week'].unique())
    historical = []
    for i, week in enumerate(weeks):
        week_df = df_season[df_season['week'] == week].copy()
        if len(week_df) <= 1:
            break
        if i + 1 < len(weeks):
            next_contestants = set(df_season[df_season['week'] == weeks[i+1]]['celebrity_name'])
            eliminated = set(week_df['celebrity_name']) - next_contestants
            elim_name = list(eliminated)[0] if eliminated else None
        else:
            elim_name = None
        historical.append({'week': week, 'week_idx': i, 'eliminated': elim_name, 
                          'week_data': week_df, 'n': len(week_df)})
    return historical


# ============================================================================
# 简化版反推粉丝投票
# ============================================================================

def inverse_fan_vote_simple(week_data, eliminated, lam):
    """
    简化版反推：给定 λ 和淘汰者，计算能解释淘汰的粉丝投票
    
    综合得分 S = λ*J + (1-λ)*F，被淘汰者 S 最低
    反推 F 使得淘汰者的 S 最低
    """
    names = week_data['celebrity_name'].tolist()
    n = len(names)
    J = week_data['judge_total'].values.astype(float)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
    
    if eliminated is None or lam >= 0.99:
        return {nm: j_share[i] for i, nm in enumerate(names)}
    
    # 找到被淘汰者索引
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    if eliminated not in name_to_idx:
        return {nm: j_share[i] for i, nm in enumerate(names)}
    
    elim_idx = name_to_idx[eliminated]
    
    # 构造粉丝投票使淘汰者综合得分最低
    # S_elim = λ*j_elim + (1-λ)*f_elim 要最小
    # 对于其他人 S_i = λ*j_i + (1-λ)*f_i 要更大
    
    # 简单策略：给淘汰者最低粉丝份额，其他人按评委比例分配剩余
    f = j_share.copy()
    
    # 降低淘汰者的粉丝投票
    min_share = 0.01 / n  # 给一个很小的值
    reduction = f[elim_idx] - min_share
    if reduction > 0:
        f[elim_idx] = min_share
        # 把减少的部分分配给其他人
        other_idx = [i for i in range(n) if i != elim_idx]
        for i in other_idx:
            f[i] += reduction / len(other_idx)
    
    # 归一化
    f = f / f.sum()
    
    return {nm: f[i] for i, nm in enumerate(names)}


# ============================================================================
# 计算指标
# ============================================================================

def simulate_elimination(df_season, lambda_vec, regime='percent'):
    """模拟淘汰"""
    weeks = sorted(df_season['week'].unique())
    all_contestants = df_season['celebrity_name'].unique().tolist()
    survival_weeks = {name: 0 for name in all_contestants}
    alive = set(all_contestants)
    
    for i, week in enumerate(weeks):
        lam = lambda_vec[i] if i < len(lambda_vec) else 0.5
        week_df = df_season[(df_season['week'] == week) & (df_season['celebrity_name'].isin(alive))].copy()
        
        if len(week_df) <= 1:
            for name in alive:
                survival_weeks[name] = i + 1
            break
        
        week_df['judge_share'] = week_df['judge_total'] / week_df['judge_total'].sum()
        week_df['fan_share'] = week_df['fan_vote_share'] / week_df['fan_vote_share'].sum()
        week_df['combined'] = lam * week_df['judge_share'] + (1 - lam) * week_df['fan_share']
        eliminated = week_df.loc[week_df['combined'].idxmin(), 'celebrity_name']
        
        for name in alive:
            survival_weeks[name] = i + 1
        alive.remove(eliminated)
    
    for name in alive:
        survival_weeks[name] = len(weeks)
    
    return survival_weeks


def compute_metrics(df_season, df_trends, schedule, season, lambda_vec, regime='percent'):
    """计算五个指标"""
    # 模拟淘汰
    survival_weeks = simulate_elimination(df_season, lambda_vec, regime)
    
    # Celebrity
    celebrity = sum((df_season[df_season['celebrity_name']==name]['Celebrity_Average_Popularity_Score'].iloc[0] 
                    if len(df_season[df_season['celebrity_name']==name]) > 0 else 0) * weeks 
                   for name, weeks in survival_weeks.items())
    
    # Dancer
    dancer = sum((df_season[df_season['celebrity_name']==name]['ballroom_partner_Average_Popularity_Score'].iloc[0]
                 if len(df_season[df_season['celebrity_name']==name]) > 0 else 0) * weeks
                for name, weeks in survival_weeks.items())
    
    # CCVI
    if season in schedule:
        start, end = schedule[season]
        mask = (df_trends['Time'] >= pd.to_datetime(start)) & (df_trends['Time'] <= pd.to_datetime(end))
        data = df_trends[mask]['search_volume'].values
        ccvi = 0.3 * np.max(data) + 0.5 * np.sum(data) + 0.2 * max(0, np.max(np.diff(data))) if len(data) > 1 else 0
    else:
        ccvi = 0
    
    # Satisfaction
    stats = []
    for name, weeks in survival_weeks.items():
        fan_pcts = []
        for w in df_season[df_season['celebrity_name']==name]['week'].unique():
            week_all = df_season[df_season['week']==w]
            if len(week_all) > 0:
                fvs = df_season[(df_season['week']==w) & (df_season['celebrity_name']==name)]['fan_vote_share'].iloc[0]
                rank = (week_all['fan_vote_share'] > fvs).sum() + 1
                fan_pcts.append(rank / len(week_all))
        if fan_pcts:
            stats.append({'mean_pct': np.mean(fan_pcts), 'weeks': weeks})
    
    if len(stats) >= 3:
        stats_df = pd.DataFrame(stats)
        stats_df['placement'] = stats_df['weeks'].rank(ascending=False) / len(stats_df)
        satisfaction, _ = spearmanr(stats_df['mean_pct'], stats_df['placement'])
        satisfaction = satisfaction if not np.isnan(satisfaction) else 0
    else:
        satisfaction = 0
    
    # Discrepancy - 基于反推粉丝投票
    historical = extract_historical_elimination(df_season)
    total_disc = 0
    for h in historical:
        if h['eliminated'] is None:
            continue
        week_idx = h['week_idx']
        lam = lambda_vec[week_idx] if week_idx < len(lambda_vec) else 0.5
        week_data = h['week_data']
        n = h['n']
        
        # 反推粉丝投票
        fan_dict = inverse_fan_vote_simple(week_data, h['eliminated'], lam)
        
        names = week_data['celebrity_name'].tolist()
        fan_shares = np.array([fan_dict[nm] for nm in names])
        fan_ranks = rankdata(-fan_shares, method='average') / n
        
        judge_scores = week_data['judge_total'].values
        judge_ranks = rankdata(-judge_scores, method='average') / n
        
        week_disc = np.sum(np.abs(fan_ranks - judge_ranks)) * n
        total_disc += week_disc
    
    return {'celebrity': celebrity, 'dancer': dancer, 'ccvi': ccvi, 
            'satisfaction': satisfaction, 'discrepancy': total_disc}


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("Task 4: 动态权重优化 V5 (快速版)")
    print("λ ∈ {0.2, 0.4, 0.6, 0.8}，5个代表性赛季")
    print("收益 = w1*Celeb + w2*Dancer + w3*CCVI + w4*Satisf - w5*Discrepancy")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据...")
    df_full, df_fan, df_trends, schedule = load_data()
    
    # 选择的赛季：3个percent + 2个bottom2
    test_seasons = [
        (3, 'percent'), (10, 'percent'), (20, 'percent'),
        (28, 'bottom2'), (32, 'bottom2')
    ]
    
    # 权重
    weights = (0.2, 0.1, 0.3, 0.2, 0.2)
    lambda_options = [0.2, 0.4, 0.6, 0.8]
    
    # 预热：计算标准化参数
    print("\n预热阶段...")
    all_raw = []
    for season, regime in test_seasons:
        df_season = df_fan[df_fan['season'] == season].copy()
        df_season['Celebrity_Average_Popularity_Score'].fillna(0, inplace=True)
        df_season['ballroom_partner_Average_Popularity_Score'].fillna(0, inplace=True)
        n_weeks = df_season['week'].nunique()
        metrics = compute_metrics(df_season, df_trends, schedule, season, [0.5]*n_weeks, regime)
        all_raw.append(metrics)
        print(f"  Season {season}: celeb={metrics['celebrity']:.0f}, disc={metrics['discrepancy']:.2f}")
    
    # 计算标准化参数
    scale_params = {}
    for key in ['celebrity', 'dancer', 'ccvi', 'satisfaction', 'discrepancy']:
        vals = [m[key] for m in all_raw]
        mean, std = np.mean(vals), np.std(vals)
        z_vals = (np.array(vals) - mean) / std if std > 0 else np.zeros(len(vals))
        scale_params[key] = {'mean': mean, 'std': std, 'z_min': z_vals.min(), 'z_max': z_vals.max()}
    
    def normalize(val, key):
        p = scale_params[key]
        if p['std'] == 0:
            return 0.5
        z = (val - p['mean']) / p['std']
        if p['z_max'] == p['z_min']:
            return 0.5
        return np.clip((z - p['z_min']) / (p['z_max'] - p['z_min']), 0, 1)
    
    def compute_benefit(metrics):
        c = normalize(metrics['celebrity'], 'celebrity')
        d = normalize(metrics['dancer'], 'dancer')
        v = normalize(metrics['ccvi'], 'ccvi')
        s = normalize(metrics['satisfaction'], 'satisfaction')
        disc = normalize(metrics['discrepancy'], 'discrepancy')
        # Discrepancy 取负值
        return weights[0]*c + weights[1]*d + weights[2]*v + weights[3]*s - weights[4]*disc
    
    # 搜索
    print("\n" + "=" * 70)
    print("搜索最优 λ 向量")
    print("=" * 70)
    
    results = []
    
    for season, regime in test_seasons:
        df_season = df_fan[df_fan['season'] == season].copy()
        df_season['Celebrity_Average_Popularity_Score'].fillna(0, inplace=True)
        df_season['ballroom_partner_Average_Popularity_Score'].fillna(0, inplace=True)
        n_weeks = df_season['week'].nunique()
        
        total_comb = len(lambda_options) ** n_weeks
        print(f"\nSeason {season} ({regime}, {n_weeks}周, {total_comb}种组合)")
        
        # 基准
        baseline = compute_metrics(df_season, df_trends, schedule, season, [0.5]*n_weeks, regime)
        baseline_benefit = compute_benefit(baseline)
        
        # 搜索
        best_lambda = None
        best_benefit = -np.inf
        best_metrics = None
        
        start_time = time.time()
        count = 0
        
        for lambda_vec in product(lambda_options, repeat=n_weeks):
            lambda_vec = list(lambda_vec)
            metrics = compute_metrics(df_season, df_trends, schedule, season, lambda_vec, regime)
            benefit = compute_benefit(metrics)
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_lambda = lambda_vec
                best_metrics = metrics
            
            count += 1
            if count % 1000 == 0:
                print(f"  已搜索 {count}/{total_comb}...", end='\r')
        
        elapsed = time.time() - start_time
        improvement = best_benefit - baseline_benefit
        
        print(f"  完成! 用时 {elapsed:.1f}s")
        print(f"  最优λ: {best_lambda}")
        print(f"  收益: {best_benefit:.4f} (基准: {baseline_benefit:.4f}, 改进: {improvement:+.4f})")
        print(f"  Discrepancy: {best_metrics['discrepancy']:.2f}")
        
        results.append({
            'season': season, 'regime': regime, 'n_weeks': n_weeks,
            'optimal_lambda': best_lambda, 'lambda_mean': np.mean(best_lambda),
            'benefit': best_benefit, 'baseline': baseline_benefit, 'improvement': improvement,
            **{f'{k}_raw': v for k, v in best_metrics.items()}
        })
    
    # 保存
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print(results_df[['season', 'regime', 'lambda_mean', 'benefit', 'improvement']].to_string())
    
    results_df.to_csv(r'd:\2026-repo\data\task4_results_v5.csv', index=False)
    print(f"\n结果已保存到 data/task4_results_v5.csv")


if __name__ == "__main__":
    main()
