# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 V4 Fast

使用差分进化算法快速搜索最优 λ 向量
- 每周 λ ∈ [0.2, 0.8]
- 只跑5个代表赛季
- Discrepancy 取负值

Author: MCM 2026
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.optimize import differential_evolution
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 数据加载
# ============================================================================

def load_data():
    """加载数据"""
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
    
    df_fan = pd.read_csv(r'd:\2026-repo\data\fan_vote_results_final.csv')
    pop_cols = ['season', 'week', 'celebrity_name', 'Celebrity_Average_Popularity_Score', 
                'ballroom_partner_Average_Popularity_Score', 'placement']
    df_fan = df_fan.merge(df_full[pop_cols].drop_duplicates(subset=['season', 'week', 'celebrity_name']),
                          on=['season', 'week', 'celebrity_name'], how='left')
    
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
        
        if i + 1 < len(weeks):
            next_contestants = set(df_season[df_season['week'] == weeks[i + 1]]['celebrity_name'])
            current = set(week_df['celebrity_name'])
            eliminated = list(current - next_contestants)
            eliminated_name = eliminated[0] if eliminated else None
        else:
            eliminated_name = None
        
        results.append({
            'week': week, 'week_idx': i,
            'eliminated': eliminated_name,
            'week_data': week_df,
            'n': len(week_df)
        })
    
    return results


# ============================================================================
# 模拟淘汰
# ============================================================================

def simulate_elimination(df_season, lambda_vec, regime='percent'):
    """模拟淘汰过程"""
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
        
        if regime == 'percent':
            j_sum = week_df['judge_total'].sum()
            f_sum = week_df['fan_vote_share'].sum()
            week_df['score'] = lam * (week_df['judge_total'] / j_sum) + (1 - lam) * (week_df['fan_vote_share'] / f_sum)
            eliminated = week_df.loc[week_df['score'].idxmin(), 'celebrity_name']
        else:
            week_df['j_rank'] = week_df['judge_total'].rank(ascending=False)
            week_df['f_rank'] = week_df['fan_vote_share'].rank(ascending=False)
            week_df['rank'] = lam * week_df['j_rank'] + (1 - lam) * week_df['f_rank']
            eliminated = week_df.loc[week_df['rank'].idxmax(), 'celebrity_name']
        
        for name in alive:
            survival_weeks[name] = i + 1
        alive.remove(eliminated)
    
    for name in alive:
        survival_weeks[name] = len(weeks)
    
    return survival_weeks


# ============================================================================
# 计算指标
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


def compute_discrepancy(historical_results, lambda_vec, regime='percent'):
    """
    计算分歧度：反推粉丝投票，计算 |fan_rank_pct - judge_rank_pct|
    """
    total = 0
    
    for result in historical_results:
        week_data = result['week_data']
        eliminated = result['eliminated']
        week_idx = result['week_idx']
        n = result['n']
        
        if eliminated is None or n == 0:
            continue
        
        lam = lambda_vec[week_idx] if week_idx < len(lambda_vec) else 0.5
        
        # 简化反推：根据淘汰约束估计粉丝投票
        names = week_data['celebrity_name'].tolist()
        J = week_data['judge_total'].values.astype(float)
        j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
        
        # 反推粉丝投票：被淘汰者的综合得分最低
        # combined = lam * j_share + (1-lam) * f_share
        # 被淘汰者 combined 最小 → f_share 需要足够低
        elim_idx = names.index(eliminated) if eliminated in names else -1
        
        if elim_idx >= 0 and lam < 0.99:
            # 简化反推：让被淘汰者粉丝份额最低
            f_share = j_share.copy()
            # 调整使被淘汰者综合得分最低
            min_j = j_share[elim_idx]
            others_j = [j_share[i] for i in range(n) if i != elim_idx]
            
            # 被淘汰者粉丝份额设为最低
            f_share[elim_idx] = 0.01
            remaining = 1 - 0.01
            for i in range(n):
                if i != elim_idx:
                    f_share[i] = remaining / (n - 1)
        else:
            f_share = j_share.copy()
        
        # 计算排名百分比
        fan_ranks = rankdata(-f_share, method='average') / n
        judge_ranks = rankdata(-J, method='average') / n
        
        # 分歧度
        week_disc = np.sum(np.abs(fan_ranks - judge_ranks)) * n
        total += week_disc
    
    return total


# ============================================================================
# 总收益计算
# ============================================================================

class SeasonOptimizer:
    """单个赛季的优化器"""
    
    def __init__(self, df_season, df_trends, schedule, season, scale_params, weights, regime):
        self.df_season = df_season
        self.df_trends = df_trends
        self.schedule = schedule
        self.season = season
        self.scale_params = scale_params
        self.weights = weights
        self.regime = regime
        self.n_weeks = df_season['week'].nunique()
        self.historical = extract_historical_elimination(df_season)
        self.eval_count = 0
        
        # 预计算不变的指标
        self.ccvi = compute_ccvi(df_trends, schedule, season)
    
    def normalize(self, val, col):
        p = self.scale_params[col]
        if p['std'] == 0:
            return 0.5
        z = (val - p['mean']) / p['std']
        if p['z_max'] == p['z_min']:
            return 0.5
        return np.clip((z - p['z_min']) / (p['z_max'] - p['z_min']), 0, 1)
    
    def objective(self, lambda_vec):
        """目标函数：返回负收益（用于最小化）"""
        self.eval_count += 1
        
        survival = simulate_elimination(self.df_season, lambda_vec, self.regime)
        
        celebrity = compute_celebrity_benefit(self.df_season, survival)
        dancer = compute_dancer_effect(self.df_season, survival)
        satisfaction = compute_satisfaction(self.df_season, survival)
        discrepancy = compute_discrepancy(self.historical, lambda_vec, self.regime)
        
        w1, w2, w3, w4, w5 = self.weights
        
        c_norm = self.normalize(celebrity, 'celebrity')
        d_norm = self.normalize(dancer, 'dancer')
        ccvi_norm = self.normalize(self.ccvi, 'ccvi')
        s_norm = self.normalize(satisfaction, 'satisfaction')
        disc_norm = self.normalize(discrepancy, 'discrepancy')
        
        # 总收益：Discrepancy 取负值
        benefit = w1 * c_norm + w2 * d_norm + w3 * ccvi_norm + w4 * s_norm - w5 * disc_norm
        
        return -benefit  # 返回负值用于最小化
    
    def optimize(self, maxiter=50):
        """使用差分进化优化"""
        bounds = [(0.2, 0.8)] * self.n_weeks
        
        result = differential_evolution(
            self.objective, bounds,
            seed=42, maxiter=maxiter, tol=0.01,
            mutation=(0.5, 1), recombination=0.7,
            polish=False, disp=False, workers=1
        )
        
        best_lambda = result.x
        best_benefit = -result.fun
        
        return best_lambda, best_benefit, self.eval_count


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("Task 4: 动态权重优化系统 V4 Fast")
    print("使用差分进化算法，每周 λ ∈ [0.2, 0.8]")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    df_full, df_fan, df_trends, schedule = load_data()
    
    # 选择5个代表赛季
    # percent: 5, 8, 12, 18, 25 (不同时期)
    test_seasons = [5, 8, 12, 18, 25]
    print(f"    测试赛季: {test_seasons}")
    
    weights = (0.2, 0.1, 0.3, 0.2, 0.2)
    print(f"\n企业权重: w1=0.2, w2=0.1, w3=0.3, w4=0.2, w5=0.2")
    print("收益公式: w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy")
    
    # 预热：用全部赛季 λ=0.5 计算归一化参数
    print("\n[2/4] 预热阶段...")
    all_metrics = {'celebrity': [], 'dancer': [], 'ccvi': [], 'satisfaction': [], 'discrepancy': []}
    
    for season in sorted(df_fan['season'].unique()):
        df_s = df_fan[df_fan['season'] == season].copy()
        df_s['Celebrity_Average_Popularity_Score'] = df_s['Celebrity_Average_Popularity_Score'].fillna(0)
        df_s['ballroom_partner_Average_Popularity_Score'] = df_s['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        n_weeks = df_s['week'].nunique()
        regime = df_s['regime'].iloc[0] if 'regime' in df_s.columns else 'percent'
        
        lambda_vec = [0.5] * n_weeks
        survival = simulate_elimination(df_s, lambda_vec, regime)
        historical = extract_historical_elimination(df_s)
        
        all_metrics['celebrity'].append(compute_celebrity_benefit(df_s, survival))
        all_metrics['dancer'].append(compute_dancer_effect(df_s, survival))
        all_metrics['ccvi'].append(compute_ccvi(df_trends, schedule, season))
        all_metrics['satisfaction'].append(compute_satisfaction(df_s, survival))
        all_metrics['discrepancy'].append(compute_discrepancy(historical, lambda_vec, regime))
    
    scale_params = {}
    for col in all_metrics:
        vals = np.array(all_metrics[col])
        mean, std = np.mean(vals), np.std(vals)
        z = (vals - mean) / std if std > 0 else np.zeros_like(vals)
        scale_params[col] = {'mean': mean, 'std': std, 'z_min': np.min(z), 'z_max': np.max(z)}
        print(f"    {col}: mean={mean:.2f}, std={std:.2f}")
    
    # 优化
    print("\n[3/4] 优化搜索...")
    print("=" * 70)
    
    results = []
    
    for season in test_seasons:
        df_s = df_fan[df_fan['season'] == season].copy()
        df_s['Celebrity_Average_Popularity_Score'] = df_s['Celebrity_Average_Popularity_Score'].fillna(0)
        df_s['ballroom_partner_Average_Popularity_Score'] = df_s['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        n_weeks = df_s['week'].nunique()
        regime = df_s['regime'].iloc[0] if 'regime' in df_s.columns else 'percent'
        
        print(f"\nSeason {season} ({regime}, {n_weeks}周)...")
        start_time = time.time()
        
        optimizer = SeasonOptimizer(df_s, df_trends, schedule, season, scale_params, weights, regime)
        best_lambda, best_benefit, eval_count = optimizer.optimize(maxiter=50)
        
        elapsed = time.time() - start_time
        
        # 基准
        baseline_lambda = [0.5] * n_weeks
        baseline_benefit = -optimizer.objective(baseline_lambda)
        improvement = best_benefit - baseline_benefit
        
        print(f"    耗时: {elapsed:.1f}s, 评估次数: {eval_count}")
        print(f"    最优λ向量: [{', '.join([f'{x:.2f}' for x in best_lambda])}]")
        print(f"    平均λ: {np.mean(best_lambda):.3f}")
        print(f"    收益: {best_benefit:.4f} (基准: {baseline_benefit:.4f}, 改进: {improvement:+.4f})")
        
        results.append({
            'season': season, 'regime': regime, 'n_weeks': n_weeks,
            'optimal_lambda_vec': list(best_lambda),
            'optimal_lambda_mean': np.mean(best_lambda),
            'benefit': best_benefit,
            'baseline': baseline_benefit,
            'improvement': improvement
        })
    
    # 保存
    print("\n[4/4] 保存结果...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(r'd:\2026-repo\data\task4_optimal_results_v4_fast.csv', index=False)
    
    print("\n" + "=" * 70)
    print("优化结果汇总")
    print("=" * 70)
    print(f"\n平均最优λ: {results_df['optimal_lambda_mean'].mean():.3f}")
    print(f"平均收益: {results_df['benefit'].mean():.4f}")
    print(f"平均改进: {results_df['improvement'].mean():.4f}")
    
    print("\n按赛制分组:")
    print(results_df.groupby('regime')[['optimal_lambda_mean', 'benefit', 'improvement']].mean().round(4))
    
    print(f"\n完成！结果保存至 data/task4_optimal_results_v4_fast.csv")


if __name__ == "__main__":
    main()
