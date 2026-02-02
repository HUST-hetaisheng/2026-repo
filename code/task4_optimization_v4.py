# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 V4

核心设计：
1. λ 是每周动态变化的向量 [λ1, λ2, ..., λn]
2. 每周 λ ∈ {0.1, 0.2, ..., 0.9}，共9种选择
3. 搜索空间巨大（9^n），使用启发式优化算法
4. 总收益 = w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy
   （Discrepancy 取负值，分歧度越小越好）
5. Discrepancy 基于反推的粉丝投票计算 |fan_rank_pct - judge_rank_pct|

Author: MCM 2026
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.optimize import minimize, linprog, differential_evolution
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 从 fan_vote_inverse_final.py 融入的反演核心函数
# ============================================================================

def feasible_f_percent(j_share, elim_idx, non_elim_idx):
    """使用线性规划找到满足淘汰约束的可行粉丝投票份额"""
    n = len(j_share)
    A_ub, b_ub = [], []
    for e in elim_idx:
        for i in non_elim_idx:
            delta = j_share[e] - j_share[i]
            row = np.zeros(n)
            row[i] = -1
            row[e] = 1
            A_ub.append(row)
            b_ub.append(-delta)
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n
    res = linprog(c=np.zeros(n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return (res.x, True) if res.success else (None, False)


def solve_percent_week_for_sim(J, names, eliminated, placement_map, f_prev=None, lam1=1.0, lam2=0.5, rho=0.5):
    """对 percent 赛制，求解满足淘汰约束的粉丝投票份额"""
    n = len(J)
    J = np.array(J, dtype=float)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
    if f_prev is None or len(f_prev) != n:
        f_prev = j_share.copy()

    pairs = []
    if len(eliminated) > 1:
        for a in eliminated:
            for b in eliminated:
                if a != b:
                    pa, pb = placement_map.get(a), placement_map.get(b)
                    if pd.notna(pa) and pd.notna(pb) and pa > pb:
                        pairs.append((a, b))

    def obj(f):
        loss = lam1 * np.sum((f - f_prev) ** 2) + lam2 * np.sum((f - j_share) ** 2)
        if pairs:
            S = j_share + f
            name_to_idx = {nm: i for i, nm in enumerate(names)}
            for a, b in pairs:
                loss += rho * max(0.0, S[name_to_idx[a]] - S[name_to_idx[b]])
        return loss

    cons = [{"type": "eq", "fun": lambda f: np.sum(f) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    if eliminated:
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
        non_elim_idx = [i for i in range(n) if i not in elim_idx]
        for e in elim_idx:
            for i in non_elim_idx:
                def ineq(f, e_idx=e, i_idx=i):
                    S = j_share + f
                    return S[i_idx] - S[e_idx]
                cons.append({"type": "ineq", "fun": ineq})

    x0 = np.clip((f_prev + j_share) / 2, 1e-6, 1.0)
    x0 = x0 / x0.sum()

    if eliminated:
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
        non_elim_idx = [i for i in range(n) if i not in elim_idx]
        f_feas, ok = feasible_f_percent(j_share, elim_idx, non_elim_idx)
        if ok:
            x0 = f_feas

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9, "maxiter": 800})
    if not res.success:
        f = x0
    else:
        f = np.clip(res.x, 0, None)
        f = f / f.sum() if f.sum() > 0 else np.ones(n) / n
    return f


def enforce_elimination_rank(names, J, eliminated, fan_ranks):
    """确保被淘汰者在综合排名中是最差的（排名制）"""
    if not eliminated:
        return fan_ranks
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
    j_rank = rankdata(-J, method="dense")
    k = len(elim_idx)
    for _ in range(200):
        R = j_rank + fan_ranks
        worst_idx = list(np.argsort(-R)[:k])
        if set(elim_idx).issubset(set(worst_idx)):
            break
        offenders = [i for i in worst_idx if i not in elim_idx]
        missing = [i for i in elim_idx if i not in worst_idx]
        if not offenders or not missing:
            break
        fan_ranks[offenders[0]], fan_ranks[missing[0]] = fan_ranks[missing[0]], fan_ranks[offenders[0]]
    return fan_ranks


def enforce_elimination_bottom2(names, J, eliminated, fan_ranks):
    """确保被淘汰者在底部2名中（bottom2制）"""
    if not eliminated:
        return fan_ranks
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
    j_rank = rankdata(-J, method="dense")
    k = len(elim_idx)
    for _ in range(200):
        R = j_rank + fan_ranks
        if k == 1:
            bottom2 = list(np.argsort(-R)[:2])
            if elim_idx[0] in bottom2:
                break
            fan_ranks[bottom2[0]], fan_ranks[elim_idx[0]] = fan_ranks[elim_idx[0]], fan_ranks[bottom2[0]]
        else:
            worst_idx = list(np.argsort(-R)[:k])
            if set(elim_idx).issubset(set(worst_idx)):
                break
            offenders = [i for i in worst_idx if i not in elim_idx]
            missing = [i for i in elim_idx if i not in worst_idx]
            if not offenders or not missing:
                break
            fan_ranks[offenders[0]], fan_ranks[missing[0]] = fan_ranks[missing[0]], fan_ranks[offenders[0]]
    return fan_ranks


def assign_fan_ranks_for_sim(names, J, eliminated, placement_map, regime, prev_rank_map=None, gamma=0.3):
    """对 rank 和 bottom2 赛制，分配满足淘汰约束的粉丝排名"""
    n = len(names)
    j_rank = rankdata(-J, method="dense")
    pref_order = sorted(range(n), key=lambda i: prev_rank_map.get(names[i], j_rank[i])) if prev_rank_map else list(np.argsort(j_rank))
    
    fan_ranks = np.zeros(n, dtype=int)
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    
    if eliminated:
        elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
        def place_key(idx):
            p = placement_map.get(names[idx])
            return p if pd.notna(p) else 1e9
        elim_idx_sorted = sorted(elim_idx, key=place_key, reverse=True)
        worst_ranks = list(range(n - len(elim_idx) + 1, n + 1))
        for r, idx in zip(worst_ranks, elim_idx_sorted):
            fan_ranks[idx] = r

    used = set(fan_ranks[fan_ranks > 0])
    for idx in pref_order:
        if fan_ranks[idx] > 0:
            continue
        for r in range(1, n + 1):
            if r not in used:
                fan_ranks[idx] = r
                used.add(r)
                break

    fan_ranks = enforce_elimination_rank(names, J, eliminated, fan_ranks) if regime == "rank" else enforce_elimination_bottom2(names, J, eliminated, fan_ranks)
    f = np.exp(-gamma * (fan_ranks - 1))
    return f / f.sum()


def inverse_fan_vote_for_simulation(week_data, eliminated_name, lam, regime='percent', f_prev=None, placement_map=None):
    """给定淘汰结果和 λ，反推满足该淘汰的粉丝投票份额"""
    names = week_data['celebrity_name'].tolist()
    n = len(names)
    J = week_data['judge_total'].values.astype(float)
    
    if placement_map is None:
        placement_map = {}
    eliminated = [eliminated_name] if eliminated_name else []
    
    if regime == 'percent':
        if lam >= 0.999:
            return {nm: 1.0/n for nm in names}
        if f_prev is None:
            j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
            f_prev_arr = j_share.copy()
        else:
            f_prev_arr = np.array([f_prev.get(nm, 1.0/n) for nm in names])
            f_prev_arr = f_prev_arr / f_prev_arr.sum()
        f = solve_percent_week_for_sim(J, names, eliminated, placement_map, f_prev=f_prev_arr, lam1=1.0, lam2=0.5, rho=0.5)
        return {nm: f[i] for i, nm in enumerate(names)}
    else:
        f = assign_fan_ranks_for_sim(names, J, eliminated, placement_map, regime, prev_rank_map=None, gamma=0.3)
        return {nm: f[i] for i, nm in enumerate(names)}


# ============================================================================
# 数据加载
# ============================================================================

def interpolate_missing_popularity(df, col_name):
    """使用均值填充人气数据中的0值"""
    df = df.copy()
    for season in df['season'].unique():
        season_mask = df['season'] == season
        season_df = df[season_mask]
        for contestant in season_df['celebrity_name'].unique():
            contestant_mask = season_mask & (df['celebrity_name'] == contestant)
            values = df.loc[contestant_mask, col_name].values
            non_zero_values = values[values > 0]
            if len(non_zero_values) == 0:
                global_mean = df[df[col_name] > 0][col_name].mean()
                df.loc[contestant_mask, col_name] = global_mean
            elif len(non_zero_values) < len(values):
                df.loc[contestant_mask & (df[col_name] == 0), col_name] = non_zero_values.mean()
    return df


def load_data():
    """加载原始数据"""
    df_full = pd.read_csv(r'd:\2026-repo\data\task3_dataset_full.csv')
    df_full = interpolate_missing_popularity(df_full, 'Celebrity_Average_Popularity_Score')
    df_full = interpolate_missing_popularity(df_full, 'ballroom_partner_Average_Popularity_Score')
    
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
# 提取历史淘汰结果
# ============================================================================

def extract_historical_elimination(df_season, regime='percent'):
    """从原始数据中提取历史真实淘汰结果"""
    weeks = sorted(df_season['week'].unique())
    all_contestants = df_season['celebrity_name'].unique().tolist()
    
    survival_weeks = {name: len(df_season[df_season['celebrity_name'] == name]['week'].unique()) 
                      for name in all_contestants}
    
    historical_results = []
    for i, week in enumerate(weeks):
        week_df = df_season[df_season['week'] == week].copy()
        if len(week_df) <= 1:
            break
        
        if i + 1 < len(weeks):
            next_week_contestants = set(df_season[df_season['week'] == weeks[i + 1]]['celebrity_name'].tolist())
            eliminated_this_week = set(week_df['celebrity_name'].tolist()) - next_week_contestants
            eliminated_name = list(eliminated_this_week)[0] if eliminated_this_week else None
        else:
            eliminated_name = None
        
        historical_results.append({
            'week': week, 'week_idx': i, 'n_contestants': len(week_df),
            'eliminated': eliminated_name, 'week_data': week_df.copy(), 'regime': regime
        })
    
    return survival_weeks, historical_results


# ============================================================================
# 模拟淘汰过程
# ============================================================================

def simulate_elimination(df_season, lambda_vec, regime='percent'):
    """给定每周 λ 向量，模拟淘汰过程"""
    weeks = sorted(df_season['week'].unique())
    n_weeks = len(weeks)
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
            week_df['judge_share'] = week_df['judge_total'] / week_df['judge_total'].sum()
            week_df['fan_share'] = week_df['fan_vote_share'] / week_df['fan_vote_share'].sum()
            week_df['combined_score'] = lam * week_df['judge_share'] + (1 - lam) * week_df['fan_share']
            eliminated_name = week_df.loc[week_df['combined_score'].idxmin(), 'celebrity_name']
        else:
            week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False)
            week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False)
            week_df['combined_rank'] = lam * week_df['judge_rank'] + (1 - lam) * week_df['fan_rank']
            eliminated_name = week_df.loc[week_df['combined_rank'].idxmax(), 'celebrity_name']
        
        for name in alive:
            survival_weeks[name] = i + 1
        alive.remove(eliminated_name)
    
    for name in alive:
        survival_weeks[name] = n_weeks
    
    return survival_weeks


# ============================================================================
# 计算五个指标
# ============================================================================

def compute_celebrity_benefit(df_season, survival_weeks):
    """名人收益 = Σ celebrity_popularity * survived_weeks"""
    total = 0
    for name, weeks in survival_weeks.items():
        data = df_season[df_season['celebrity_name'] == name]
        if len(data) > 0:
            pop = data['Celebrity_Average_Popularity_Score'].iloc[0]
            total += (pop if pd.notna(pop) else 0) * weeks
    return total


def compute_dancer_effect(df_season, survival_weeks):
    """舞者效应 = Σ partner_popularity * survived_weeks"""
    total = 0
    for name, weeks in survival_weeks.items():
        data = df_season[df_season['celebrity_name'] == name]
        if len(data) > 0:
            pop = data['ballroom_partner_Average_Popularity_Score'].iloc[0]
            total += (pop if pd.notna(pop) else 0) * weeks
    return total


def compute_ccvi(df_trends, schedule, season):
    """CCVI = 0.3*P + 0.5*I + 0.2*D"""
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
    return 0.3 * P + 0.5 * I + 0.2 * max(0, D)


def compute_fan_satisfaction(df_season, survival_weeks):
    """观众满意度 = Spearman(平均粉丝排名, 模拟最终名次)"""
    stats = []
    for name, weeks in survival_weeks.items():
        data = df_season[df_season['celebrity_name'] == name]
        if len(data) == 0:
            continue
        fan_rank_pcts = []
        for week in data['week'].unique():
            week_all = df_season[df_season['week'] == week]
            n = len(week_all)
            if n == 0:
                continue
            celeb_week = week_all[week_all['celebrity_name'] == name]
            if len(celeb_week) > 0:
                fvs = celeb_week['fan_vote_share'].iloc[0]
                rank = np.sum(week_all['fan_vote_share'].values > fvs) + 1
                fan_rank_pcts.append(rank / n)
        stats.append({'name': name, 'mean_fan_rank_pct': np.mean(fan_rank_pcts) if fan_rank_pcts else 0.5,
                      'survived_weeks': weeks})
    
    if len(stats) < 3:
        return 0
    stats_df = pd.DataFrame(stats)
    stats_df['simulated_placement'] = stats_df['survived_weeks'].rank(ascending=False)
    stats_df['placement_pct'] = stats_df['simulated_placement'] / len(stats_df)
    corr, _ = spearmanr(stats_df['mean_fan_rank_pct'], stats_df['placement_pct'])
    return corr if not np.isnan(corr) else 0


def compute_discrepancy_from_history(historical_results, lambda_vec):
    """
    基于历史淘汰和每周 λ 反推粉丝投票，计算分歧度
    Discrepancy = Σ |fan_rank_pct - judge_rank_pct| * n（按存活人数加权）
    """
    total_discrepancy = 0
    f_prev = None
    
    for result in historical_results:
        week_data = result['week_data']
        eliminated = result['eliminated']
        regime = result.get('regime', 'percent')
        week_idx = result.get('week_idx', 0)
        n = len(week_data)
        
        if eliminated is None or n == 0:
            continue
        
        lam = lambda_vec[week_idx] if week_idx < len(lambda_vec) else 0.5
        
        placement_map = {}
        if 'placement' in week_data.columns:
            for _, row in week_data.iterrows():
                if pd.notna(row.get('placement')):
                    placement_map[row['celebrity_name']] = row['placement']
        
        simulated_fan = inverse_fan_vote_for_simulation(week_data, eliminated, lam, regime=regime,
                                                         f_prev=f_prev, placement_map=placement_map)
        f_prev = simulated_fan
        
        names = week_data['celebrity_name'].tolist()
        fan_shares = np.array([simulated_fan.get(nm, 1.0/n) for nm in names])
        fan_ranks = rankdata(-fan_shares, method='average')
        fan_rank_pcts = fan_ranks / n
        
        judge_scores = week_data['judge_total'].values
        judge_ranks = rankdata(-judge_scores, method='average')
        judge_rank_pcts = judge_ranks / n
        
        week_discrepancy = np.sum(np.abs(fan_rank_pcts - judge_rank_pcts))
        total_discrepancy += week_discrepancy * n
    
    return total_discrepancy


# ============================================================================
# 标准化函数
# ============================================================================

def zscore_normalize(values):
    values = np.array(values, dtype=float)
    mean, std = np.nanmean(values), np.nanstd(values)
    return np.zeros_like(values) if std == 0 or np.isnan(std) else (values - mean) / std


def minmax_normalize(values):
    values = np.array(values, dtype=float)
    min_val, max_val = np.nanmin(values), np.nanmax(values)
    return np.ones_like(values) * 0.5 if max_val == min_val else (values - min_val) / (max_val - min_val)


# ============================================================================
# 主计算流程
# ============================================================================

def compute_all_metrics(df_season, df_trends, schedule, season, lambda_vec, regime='percent'):
    """计算给定 λ 向量下的所有指标"""
    # 模拟淘汰
    survival_weeks = simulate_elimination(df_season, lambda_vec, regime)
    
    # 提取历史淘汰
    _, historical_results = extract_historical_elimination(df_season, regime)
    
    celebrity = compute_celebrity_benefit(df_season, survival_weeks)
    dancer = compute_dancer_effect(df_season, survival_weeks)
    satisfaction = compute_fan_satisfaction(df_season, survival_weeks)
    ccvi = compute_ccvi(df_trends, schedule, season)
    discrepancy = compute_discrepancy_from_history(historical_results, lambda_vec)
    
    return {'celebrity': celebrity, 'dancer': dancer, 'ccvi': ccvi,
            'satisfaction': satisfaction, 'discrepancy': discrepancy,
            'survival_weeks': survival_weeks}


def compute_total_benefit(metrics, scale_params, weights):
    """
    计算总收益
    总收益 = w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy
    注意：Discrepancy 取负值（分歧度越小越好）
    """
    w1, w2, w3, w4, w5 = weights
    
    def normalize(val, col):
        mean, std = scale_params[col]['raw_mean'], scale_params[col]['raw_std']
        z_min, z_max = scale_params[col]['z_min'], scale_params[col]['z_max']
        if std == 0:
            return 0.5
        z = (val - mean) / std
        return np.clip((z - z_min) / (z_max - z_min), 0, 1) if z_max != z_min else 0.5
    
    celeb_norm = normalize(metrics['celebrity'], 'celebrity')
    dancer_norm = normalize(metrics['dancer'], 'dancer')
    ccvi_norm = normalize(metrics['ccvi'], 'ccvi')
    satisfaction_norm = normalize(metrics['satisfaction'], 'satisfaction')
    discrepancy_norm = normalize(metrics['discrepancy'], 'discrepancy')
    
    # Discrepancy 取负值：分歧度越小 → discrepancy_norm 越小 → -discrepancy_norm 越大 → 收益越高
    total = w1 * celeb_norm + w2 * dancer_norm + w3 * ccvi_norm + w4 * satisfaction_norm - w5 * discrepancy_norm
    
    return total, {'celebrity_norm': celeb_norm, 'dancer_norm': dancer_norm, 'ccvi_norm': ccvi_norm,
                   'satisfaction_norm': satisfaction_norm, 'discrepancy_norm': discrepancy_norm}


# ============================================================================
# 预热阶段
# ============================================================================

def warmup_phase(df_fan, df_trends, schedule):
    """预热阶段：用 λ=0.5 对所有赛季计算指标，获取标准化参数"""
    print("=" * 60)
    print("预热阶段：用 λ=[0.5, 0.5, ...] 计算各赛季原始指标")
    print("=" * 60)
    
    all_metrics = []
    seasons = sorted(df_fan['season'].unique())
    
    for season in seasons:
        df_season = df_fan[df_fan['season'] == season].copy()
        df_season['Celebrity_Average_Popularity_Score'] = df_season['Celebrity_Average_Popularity_Score'].fillna(0)
        df_season['ballroom_partner_Average_Popularity_Score'] = df_season['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        n_weeks = df_season['week'].nunique()
        regime = df_season['regime'].iloc[0] if 'regime' in df_season.columns else 'percent'
        
        lambda_vec = [0.5] * n_weeks
        metrics = compute_all_metrics(df_season, df_trends, schedule, season, lambda_vec, regime)
        
        all_metrics.append({'season': season, **{k: v for k, v in metrics.items() if k != 'survival_weeks'}})
        print(f"  Season {season}: celebrity={metrics['celebrity']:.1f}, discrepancy={metrics['discrepancy']:.2f}")
    
    metrics_df = pd.DataFrame(all_metrics)
    
    scale_params = {}
    for col in ['celebrity', 'dancer', 'ccvi', 'satisfaction', 'discrepancy']:
        raw_values = metrics_df[col].values
        z_values = zscore_normalize(raw_values)
        scale_params[col] = {'raw_mean': np.nanmean(raw_values), 'raw_std': np.nanstd(raw_values),
                             'z_min': np.nanmin(z_values), 'z_max': np.nanmax(z_values)}
    
    return metrics_df, scale_params


# ============================================================================
# 优化搜索（使用差分进化算法）
# ============================================================================

def search_optimal_lambda_de(df_season, df_trends, schedule, season, scale_params, weights, regime='percent'):
    """
    使用差分进化算法搜索最优 λ 向量
    搜索空间：每周 λ ∈ [0.1, 0.9]
    """
    n_weeks = df_season['week'].nunique()
    
    def objective(lambda_vec):
        metrics = compute_all_metrics(df_season, df_trends, schedule, season, lambda_vec, regime)
        benefit, _ = compute_total_benefit(metrics, scale_params, weights)
        return -benefit  # 最小化负收益 = 最大化收益
    
    bounds = [(0.2, 0.8)] * n_weeks
    
    result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-4,
                                    mutation=(0.5, 1), recombination=0.7, workers=1, 
                                    polish=False, disp=False)
    
    best_lambda_vec = result.x
    metrics = compute_all_metrics(df_season, df_trends, schedule, season, best_lambda_vec, regime)
    best_benefit, best_norms = compute_total_benefit(metrics, scale_params, weights)
    
    return best_lambda_vec, best_benefit, metrics, best_norms


def search_optimal_lambda_grid(df_season, df_trends, schedule, season, scale_params, weights, regime='percent', max_combinations=50000):
    """
    网格搜索（对于周数少的赛季）
    每周 λ ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
    """
    n_weeks = df_season['week'].nunique()
    lambda_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 7种选择
    
    total_combinations = 7 ** n_weeks
    
    if total_combinations > max_combinations:
        # 太多组合，使用差分进化
        return search_optimal_lambda_de(df_season, df_trends, schedule, season, scale_params, weights, regime)
    
    best_lambda_vec = None
    best_benefit = -np.inf
    best_metrics = None
    best_norms = None
    
    count = 0
    for lambda_vec in product(lambda_options, repeat=n_weeks):
        lambda_vec = list(lambda_vec)
        metrics = compute_all_metrics(df_season, df_trends, schedule, season, lambda_vec, regime)
        benefit, norms = compute_total_benefit(metrics, scale_params, weights)
        
        if benefit > best_benefit:
            best_benefit = benefit
            best_lambda_vec = lambda_vec
            best_metrics = metrics
            best_norms = norms
        
        count += 1
        if count % 1000 == 0:
            print(f"    已搜索 {count}/{total_combinations} ({100*count/total_combinations:.1f}%)", end="\r")
    
    return best_lambda_vec, best_benefit, best_metrics, best_norms


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("Task 4: 动态权重优化系统 V4")
    print("每周 λ 动态变化，Discrepancy 取负值")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    df_full, df_fan, df_trends, schedule = load_data()
    print(f"    粉丝投票数据: {len(df_fan)} 条记录")
    
    # 企业权重
    weights = (0.2, 0.1, 0.3, 0.2, 0.2)  # w1, w2, w3, w4, w5
    print(f"\n企业权重: w1=0.2, w2=0.1, w3=0.3, w4=0.2, w5=0.2")
    print("收益公式: w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy")
    
    # 预热阶段
    print("\n[2/4] 预热阶段...")
    metrics_df, scale_params = warmup_phase(df_fan, df_trends, schedule)
    
    # 搜索阶段
    print("\n[3/4] 搜索阶段：寻找各赛季最优 λ 向量")
    print("=" * 60)
    
    results = []
    seasons = sorted(df_fan['season'].unique())
    
    for season in seasons:
        df_season = df_fan[df_fan['season'] == season].copy()
        df_season['Celebrity_Average_Popularity_Score'] = df_season['Celebrity_Average_Popularity_Score'].fillna(0)
        df_season['ballroom_partner_Average_Popularity_Score'] = df_season['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        n_weeks = df_season['week'].nunique()
        regime = df_season['regime'].iloc[0] if 'regime' in df_season.columns else 'percent'
        
        total_combinations = 7 ** n_weeks  # 7种选择: 0.2-0.8
        method = "grid" if total_combinations <= 50000 else "DE"
        
        print(f"\n  Season {season} ({regime}, {n_weeks}周, {total_combinations:,}种组合, 方法={method})")
        
        best_lambda_vec, best_benefit, best_metrics, best_norms = search_optimal_lambda_grid(
            df_season, df_trends, schedule, season, scale_params, weights, regime)
        
        # 基准：λ=0.5
        baseline_vec = [0.5] * n_weeks
        baseline_metrics = compute_all_metrics(df_season, df_trends, schedule, season, baseline_vec, regime)
        baseline_benefit, _ = compute_total_benefit(baseline_metrics, scale_params, weights)
        
        improvement = best_benefit - baseline_benefit
        
        # 实时打印最佳 λ 向量
        lambda_str = "[" + ", ".join([f"{l:.1f}" for l in best_lambda_vec]) + "]"
        print(f"    最佳λ: {lambda_str}")
        print(f"    收益: {best_benefit:.4f} (基准: {baseline_benefit:.4f}, 改进: {improvement:+.4f})")
        print(f"    discrepancy: {best_metrics['discrepancy']:.2f}")
        
        results.append({
            'season': season, 'regime': regime, 'n_weeks': n_weeks,
            'optimal_lambda_vec': best_lambda_vec,
            'optimal_lambda_mean': np.mean(best_lambda_vec),
            'total_benefit': best_benefit,
            'baseline_benefit': baseline_benefit,
            'improvement': improvement,
            **{f'{k}_raw': v for k, v in best_metrics.items() if k != 'survival_weeks'},
            **best_norms
        })
    
    # 保存结果
    print("\n[4/4] 保存结果...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(r'd:\2026-repo\data\task4_optimal_results_v4.csv', index=False)
    
    # 统计
    print("\n" + "=" * 60)
    print("优化结果统计")
    print("=" * 60)
    print(f"\n平均最优λ: {results_df['optimal_lambda_mean'].mean():.3f}")
    print(f"平均总收益: {results_df['total_benefit'].mean():.4f}")
    print(f"平均改进: {results_df['improvement'].mean():.4f}")
    print(f"改进 > 0 的赛季数: {(results_df['improvement'] > 0).sum()} / {len(results_df)}")
    
    print("\n按赛制分组：")
    print(results_df.groupby('regime')[['optimal_lambda_mean', 'total_benefit', 'improvement']].mean().round(3))
    
    print(f"\n完成！结果保存至 data/task4_optimal_results_v4.csv")


if __name__ == "__main__":
    main()
