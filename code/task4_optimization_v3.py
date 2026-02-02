# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 V3

核心改进：
1. 所有指标统一先 Z-score 再 Min-Max 归一化到 [0,1]
2. Discrepancy 基于模拟淘汰结果反推新的粉丝投票（使用完善的反演算法）
3. CCVI 保留作为赛季基础收益
4. 正确处理原始数据（而非已标准化数据）
5. 融入 fan_vote_inverse_final.py 的完善反演逻辑

Author: MCM 2026
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.optimize import minimize, linprog
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 从 fan_vote_inverse_final.py 融入的反演核心函数
# ============================================================================

def feasible_f_percent(j_share, elim_idx, non_elim_idx):
    """
    使用线性规划找到满足淘汰约束的可行粉丝投票份额
    """
    n = len(j_share)
    A_ub = []
    b_ub = []
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
    if res.success:
        return res.x, True
    return None, False


def solve_percent_week_for_sim(J, names, eliminated, placement_map, f_prev=None,
                                lam1=1.0, lam2=0.5, rho=0.5):
    """
    对 percent 赛制，求解满足淘汰约束的粉丝投票份额
    基于 fan_vote_inverse_final.py 的 solve_percent_week
    """
    n = len(J)
    J = np.array(J, dtype=float)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n

    if f_prev is None or len(f_prev) != n:
        f_prev = j_share.copy()

    # 多淘汰者内部排序约束
    pairs = []
    if len(eliminated) > 1:
        for a in eliminated:
            for b in eliminated:
                if a == b:
                    continue
                pa = placement_map.get(a)
                pb = placement_map.get(b)
                if pd.notna(pa) and pd.notna(pb) and pa > pb:
                    pairs.append((a, b))

    def obj(f):
        loss = lam1 * np.sum((f - f_prev) ** 2) + lam2 * np.sum((f - j_share) ** 2)
        if pairs:
            S = j_share + f
            name_to_idx = {nm: i for i, nm in enumerate(names)}
            for a, b in pairs:
                ia = name_to_idx[a]
                ib = name_to_idx[b]
                loss += rho * max(0.0, S[ia] - S[ib])
        return loss

    cons = [{"type": "eq", "fun": lambda f: np.sum(f) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    # 淘汰约束：被淘汰者综合得分最低
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

    # 尝试找可行初始点
    if eliminated:
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
        non_elim_idx = [i for i in range(n) if i not in elim_idx]
        f_feas, ok = feasible_f_percent(j_share, elim_idx, non_elim_idx)
        if ok:
            x0 = f_feas

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 800})
    if not res.success:
        f = x0
    else:
        f = np.clip(res.x, 0, None)
        f = f / f.sum() if f.sum() > 0 else np.ones(n) / n
    return f


def enforce_elimination_rank(names, J, eliminated, fan_ranks):
    """
    确保被淘汰者在综合排名中是最差的（排名制）
    """
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
        i = offenders[0]
        e = missing[0]
        fan_ranks[i], fan_ranks[e] = fan_ranks[e], fan_ranks[i]
    return fan_ranks


def enforce_elimination_bottom2(names, J, eliminated, fan_ranks):
    """
    确保被淘汰者在底部2名中（bottom2制）
    """
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
            swap_idx = bottom2[0]
            e = elim_idx[0]
            fan_ranks[swap_idx], fan_ranks[e] = fan_ranks[e], fan_ranks[swap_idx]
        else:
            worst_idx = list(np.argsort(-R)[:k])
            if set(elim_idx).issubset(set(worst_idx)):
                break
            offenders = [i for i in worst_idx if i not in elim_idx]
            missing = [i for i in elim_idx if i not in worst_idx]
            if not offenders or not missing:
                break
            i = offenders[0]
            e = missing[0]
            fan_ranks[i], fan_ranks[e] = fan_ranks[e], fan_ranks[i]
    return fan_ranks


def assign_fan_ranks_for_sim(names, J, eliminated, placement_map, regime, prev_rank_map=None, gamma=0.3):
    """
    对 rank 和 bottom2 赛制，分配满足淘汰约束的粉丝排名
    返回粉丝投票份额（通过指数衰减转换）
    """
    n = len(names)
    j_rank = rankdata(-J, method="dense")

    if prev_rank_map:
        pref_order = sorted(range(n), key=lambda i: prev_rank_map.get(names[i], j_rank[i]))
    else:
        pref_order = list(np.argsort(j_rank))

    fan_ranks = np.zeros(n, dtype=int)
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    # 先为被淘汰者分配最差排名
    if eliminated:
        elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
        
        def place_key(idx):
            nm = names[idx]
            p = placement_map.get(nm)
            return p if pd.notna(p) else 1e9

        elim_idx_sorted = sorted(elim_idx, key=place_key, reverse=True)
        worst_ranks = list(range(n - len(elim_idx) + 1, n + 1))
        for r, idx in zip(worst_ranks, elim_idx_sorted):
            fan_ranks[idx] = r

    # 为其他人分配排名
    used = set(fan_ranks[fan_ranks > 0])
    for idx in pref_order:
        if fan_ranks[idx] > 0:
            continue
        for r in range(1, n + 1):
            if r not in used:
                fan_ranks[idx] = r
                used.add(r)
                break

    # 强制满足淘汰约束
    if regime == "rank":
        fan_ranks = enforce_elimination_rank(names, J, eliminated, fan_ranks)
    else:
        fan_ranks = enforce_elimination_bottom2(names, J, eliminated, fan_ranks)

    # 转换为份额（指数衰减）
    f = np.exp(-gamma * (fan_ranks - 1))
    f = f / f.sum()
    return f


# ============================================================================
# 数据加载与预处理
# ============================================================================

def interpolate_missing_popularity(df, col_name):
    """
    使用拉格朗日插值法填充人气数据中的0值（缺失值）
    按赛季分组，对每个选手的0值进行插值
    """
    df = df.copy()
    
    # 按赛季分组处理
    for season in df['season'].unique():
        season_mask = df['season'] == season
        season_df = df[season_mask]
        
        # 获取该赛季所有选手
        contestants = season_df['celebrity_name'].unique()
        
        for contestant in contestants:
            contestant_mask = season_mask & (df['celebrity_name'] == contestant)
            values = df.loc[contestant_mask, col_name].values
            
            # 如果全是0或只有一个值，使用赛季平均值
            non_zero_values = values[values > 0]
            if len(non_zero_values) == 0:
                # 使用全局非零平均值
                global_mean = df[df[col_name] > 0][col_name].mean()
                df.loc[contestant_mask, col_name] = global_mean
            elif len(non_zero_values) < len(values):
                # 有部分0值，用该选手的非零平均值填充
                contestant_mean = non_zero_values.mean()
                df.loc[contestant_mask & (df[col_name] == 0), col_name] = contestant_mean
    
    return df


def load_data():
    """加载原始数据并预处理"""
    # 主数据集 - 使用 task3_dataset_full.csv（原始版本）
    df_full = pd.read_csv(r'd:\2026-repo\data\task3_dataset_full.csv')
    
    # 对人气列的0值进行插值填充
    print("    预处理：填充人气数据中的0值...")
    df_full = interpolate_missing_popularity(df_full, 'Celebrity_Average_Popularity_Score')
    df_full = interpolate_missing_popularity(df_full, 'ballroom_partner_Average_Popularity_Score')
    
    print(f"    填充后 Celebrity 0值数量: {(df_full['Celebrity_Average_Popularity_Score']==0).sum()}")
    print(f"    填充后 Partner 0值数量: {(df_full['ballroom_partner_Average_Popularity_Score']==0).sum()}")
    
    # 粉丝投票数据 - 每周一条记录
    df_fan = pd.read_csv(r'd:\2026-repo\data\fan_vote_results_final.csv')
    
    # 从完整数据提取人气信息，合并到粉丝投票数据
    pop_cols = ['season', 'week', 'celebrity_name', 
                'Celebrity_Average_Popularity_Score', 
                'ballroom_partner_Average_Popularity_Score', 
                'placement']
    
    df_fan = df_fan.merge(
        df_full[pop_cols].drop_duplicates(subset=['season', 'week', 'celebrity_name']),
        on=['season', 'week', 'celebrity_name'],
        how='left'
    )
    
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
    
    return df_full, df_fan, df_trends, schedule


def compute_judge_total(row):
    """计算评委总分（取第一周有分数的）"""
    for w in range(1, 12):
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        scores = []
        for c in cols:
            if c in row.index and pd.notna(row[c]) and row[c] > 0:
                scores.append(row[c])
        if scores:
            return sum(scores)
    return 0


# ============================================================================
# 提取历史真实淘汰结果 / 模拟淘汰过程
# ============================================================================

def extract_historical_elimination(df_season, regime='percent'):
    """
    从原始数据中提取历史真实淘汰结果（不是模拟）
    
    返回：
    - survival_weeks: 每个选手的真实存活周数
    - historical_results: 每周的真实淘汰结果（用于反推粉丝投票）
    """
    weeks = sorted(df_season['week'].unique())
    n_weeks = len(weeks)
    
    all_contestants = df_season['celebrity_name'].unique().tolist()
    
    # 根据 placement 确定每个选手的存活周数
    survival_weeks = {}
    for name in all_contestants:
        contestant_data = df_season[df_season['celebrity_name'] == name]
        # 存活周数 = 出现的周数
        survival_weeks[name] = len(contestant_data['week'].unique())
    
    # 提取每周淘汰者
    historical_results = []
    
    for i, week in enumerate(weeks):
        week_df = df_season[df_season['week'] == week].copy()
        
        if len(week_df) <= 1:
            break
        
        # 找出这周被淘汰的选手（这周有但下周没有的）
        if i + 1 < len(weeks):
            next_week = weeks[i + 1]
            next_week_contestants = set(df_season[df_season['week'] == next_week]['celebrity_name'].tolist())
            current_contestants = set(week_df['celebrity_name'].tolist())
            eliminated_this_week = current_contestants - next_week_contestants
            
            if eliminated_this_week:
                eliminated_name = list(eliminated_this_week)[0]  # 通常每周只淘汰一人
            else:
                eliminated_name = None
        else:
            # 最后一周，没有淘汰
            eliminated_name = None
        
        historical_results.append({
            'week': week,
            'n_contestants': len(week_df),
            'eliminated': eliminated_name,
            'week_data': week_df.copy(),
            'regime': regime
        })
    
    return survival_weeks, historical_results


def simulate_elimination(df_season, lambda_vec, regime='percent'):
    """
    给定一个赛季的数据和每周评委权重λ向量，模拟淘汰过程
    
    返回：
    - survival_weeks: 每个选手的模拟存活周数
    - simulated_results: 每周的详细模拟结果（用于反推粉丝投票）
    """
    weeks = sorted(df_season['week'].unique())
    n_weeks = len(weeks)
    
    all_contestants = df_season['celebrity_name'].unique().tolist()
    survival_weeks = {name: 0 for name in all_contestants}
    alive = set(all_contestants)
    
    simulated_results = []
    
    for i, week in enumerate(weeks):
        lam = lambda_vec[i] if i < len(lambda_vec) else 0.5
        
        week_df = df_season[(df_season['week'] == week) & 
                            (df_season['celebrity_name'].isin(alive))].copy()
        
        if len(week_df) <= 1:
            for name in alive:
                survival_weeks[name] = i + 1
            break
        
        # 计算综合得分
        if regime == 'percent':
            # 百分比制：综合得分 = λ*judge_share + (1-λ)*fan_share
            week_df['judge_share'] = week_df['judge_total'] / week_df['judge_total'].sum()
            week_df['fan_share'] = week_df['fan_vote_share'] / week_df['fan_vote_share'].sum()
            week_df['combined_score'] = lam * week_df['judge_share'] + (1 - lam) * week_df['fan_share']
            # 得分最低者淘汰
            eliminated_name = week_df.loc[week_df['combined_score'].idxmin(), 'celebrity_name']
        else:
            # 排名制：综合排名 = λ*judge_rank + (1-λ)*fan_rank
            week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False)
            week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False)
            week_df['combined_rank'] = lam * week_df['judge_rank'] + (1 - lam) * week_df['fan_rank']
            # 排名最差者淘汰
            eliminated_name = week_df.loc[week_df['combined_rank'].idxmax(), 'celebrity_name']
        
        # 记录结果
        for name in alive:
            survival_weeks[name] = i + 1
        
        simulated_results.append({
            'week': week,
            'lambda': lam,
            'n_contestants': len(week_df),
            'eliminated': eliminated_name,
            'week_data': week_df.copy(),
            'regime': regime
        })
        
        alive.remove(eliminated_name)
    
    # 剩余选手存活到最后
    for name in alive:
        survival_weeks[name] = n_weeks
    
    return survival_weeks, simulated_results


# ============================================================================
# 反推粉丝投票（基于模拟淘汰结果）- 使用完善的反演算法
# ============================================================================

def inverse_fan_vote_for_simulation(week_data, eliminated_name, lam, regime='percent', 
                                     f_prev=None, placement_map=None):
    """
    给定模拟淘汰结果，反推满足该淘汰的粉丝投票份额
    
    使用 fan_vote_inverse_final.py 的完善逻辑：
    - percent 赛制：使用 SLSQP 优化带约束的目标函数
    - rank/bottom2 赛制：使用排名分配 + 指数衰减转换
    """
    names = week_data['celebrity_name'].tolist()
    n = len(names)
    J = week_data['judge_total'].values.astype(float)
    
    if placement_map is None:
        placement_map = {}
    
    eliminated = [eliminated_name] if eliminated_name else []
    
    if regime == 'percent':
        # 使用完善的百分比制反演
        if lam >= 0.999:
            # λ=1 时粉丝投票无影响，返回均匀分布
            return {nm: 1.0/n for nm in names}
        
        # 准备 f_prev
        if f_prev is None:
            j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
            f_prev_arr = j_share.copy()
        else:
            f_prev_arr = np.array([f_prev.get(nm, 1.0/n) for nm in names])
            f_prev_arr = f_prev_arr / f_prev_arr.sum()
        
        # 调用完善的求解函数
        f = solve_percent_week_for_sim(J, names, eliminated, placement_map, 
                                        f_prev=f_prev_arr, lam1=1.0, lam2=0.5, rho=0.5)
        return {nm: f[i] for i, nm in enumerate(names)}
    
    else:
        # rank 或 bottom2 赛制：使用排名分配
        f = assign_fan_ranks_for_sim(names, J, eliminated, placement_map, regime, 
                                      prev_rank_map=None, gamma=0.3)
        return {nm: f[i] for i, nm in enumerate(names)}


# ============================================================================
# 计算五个指标
# ============================================================================

def compute_celebrity_benefit(df_season, survival_weeks, celeb_pop_col='Celebrity_Average_Popularity_Score'):
    """
    名人收益 = Σ celebrity_popularity * survived_weeks
    使用原始人气分数（非Z-score）
    """
    total = 0
    for name, weeks in survival_weeks.items():
        celeb_data = df_season[df_season['celebrity_name'] == name]
        if len(celeb_data) == 0:
            continue
        # 使用Celebrity_Average_Popularity_Score（原始值）
        pop = celeb_data[celeb_pop_col].iloc[0] if celeb_pop_col in celeb_data.columns else 0
        if pd.isna(pop):
            pop = 0
        total += pop * weeks
    return total


def compute_dancer_effect(df_season, survival_weeks, partner_col='ballroom_partner_Average_Popularity_Score'):
    """
    舞者效应 = Σ partner_popularity * survived_weeks
    """
    total = 0
    for name, weeks in survival_weeks.items():
        celeb_data = df_season[df_season['celebrity_name'] == name]
        if len(celeb_data) == 0:
            continue
        partner_pop = celeb_data[partner_col].iloc[0] if partner_col in celeb_data.columns else 0
        if pd.isna(partner_pop):
            partner_pop = 0
        total += partner_pop * weeks
    return total


def compute_ccvi(df_trends, schedule, season):
    """
    CCVI = 0.3*P + 0.5*I + 0.2*D
    P: 峰值热度
    I: 累计热度
    D: 最大增量
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
    
    return 0.3 * P + 0.5 * I + 0.2 * max(0, D)


def compute_fan_satisfaction(df_season, survival_weeks):
    """
    观众满意度 = Spearman(平均粉丝排名, 模拟最终名次)
    """
    stats = []
    
    for name, weeks in survival_weeks.items():
        celeb_data = df_season[df_season['celebrity_name'] == name]
        if len(celeb_data) == 0:
            continue
        
        # 计算该选手的平均粉丝排名百分比
        fan_rank_pcts = []
        for week in celeb_data['week'].unique():
            week_all = df_season[df_season['week'] == week]
            n = len(week_all)
            if n == 0:
                continue
            celeb_week = week_all[week_all['celebrity_name'] == name]
            if len(celeb_week) > 0 and 'fan_vote_share' in celeb_week.columns:
                fvs = celeb_week['fan_vote_share'].iloc[0]
                # 在该周所有人中的排名
                all_fvs = week_all['fan_vote_share'].values
                rank = np.sum(all_fvs > fvs) + 1
                fan_rank_pcts.append(rank / n)
        
        mean_fan_rank_pct = np.mean(fan_rank_pcts) if fan_rank_pcts else 0.5
        
        stats.append({
            'name': name,
            'mean_fan_rank_pct': mean_fan_rank_pct,
            'survived_weeks': weeks
        })
    
    if len(stats) < 3:
        return 0
    
    stats_df = pd.DataFrame(stats)
    # 存活周数越多，名次越好
    stats_df['simulated_placement'] = stats_df['survived_weeks'].rank(ascending=False)
    stats_df['placement_pct'] = stats_df['simulated_placement'] / len(stats_df)
    
    corr, _ = spearmanr(stats_df['mean_fan_rank_pct'], stats_df['placement_pct'])
    return corr if not np.isnan(corr) else 0


def compute_discrepancy_from_history(historical_results, lambda_vec):
    """
    基于历史淘汰结果计算分歧度：
    
    1. 使用真实历史淘汰结果（不是模拟）
    2. 对于每周的 λ，反推"能解释这个淘汰"的粉丝投票份额
    3. 用反推的粉丝投票计算 fan_rank_pct
    4. 用原始评委评分计算 judge_rank_pct  
    5. Discrepancy = Σ |fan_rank_pct - judge_rank_pct| （按存活人数加权）
    
    分歧度越小 → λ 越合理 → 收益越高
    """
    total_discrepancy = 0
    f_prev = None  # 上一周的粉丝投票结果，用于时间连续性约束
    
    # 如果传入单一值，转换为列表
    if not isinstance(lambda_vec, (list, np.ndarray)):
        lambda_vec = [lambda_vec] * len(historical_results)
    
    for week_idx, result in enumerate(historical_results):
        week_data = result['week_data']
        eliminated = result['eliminated']
        regime = result.get('regime', 'percent')
        n = len(week_data)
        
        if eliminated is None or n == 0:
            continue
        
        # 获取当周的 λ
        lam = lambda_vec[week_idx] if week_idx < len(lambda_vec) else 0.5
        
        # 构建 placement_map（被淘汰者的名次）
        placement_map = {}
        if 'placement' in week_data.columns:
            for _, row in week_data.iterrows():
                if pd.notna(row.get('placement')):
                    placement_map[row['celebrity_name']] = row['placement']
        
        # 用当周的 λ 反推粉丝投票份额
        simulated_fan = inverse_fan_vote_for_simulation(
            week_data, eliminated, lam, 
            regime=regime, 
            f_prev=f_prev, 
            placement_map=placement_map
        )
        
        # 更新 f_prev 供下一周使用
        f_prev = simulated_fan
        
        # 计算反推粉丝投票的排名百分比
        names = week_data['celebrity_name'].tolist()
        fan_shares = np.array([simulated_fan.get(nm, 1.0/n) for nm in names])
        fan_ranks = rankdata(-fan_shares, method='average')  # 份额越高排名越靠前
        fan_rank_pcts = fan_ranks / n
        
        # 计算评委评分的排名百分比
        judge_scores = week_data['judge_total'].values
        judge_ranks = rankdata(-judge_scores, method='average')  # 分数越高排名越靠前
        judge_rank_pcts = judge_ranks / n
        
        # 周分歧度 = Σ |fan_rank_pct - judge_rank_pct|，按存活人数加权
        week_discrepancy = np.sum(np.abs(fan_rank_pcts - judge_rank_pcts))
        total_discrepancy += week_discrepancy * n  # 按存活人数加权
    
    return total_discrepancy


def compute_discrepancy_simple(df_season):
    """
    简化版分歧度：评委与粉丝排名差异
    Σ |fan_rank_pct - judge_rank_pct|
    """
    weeks = sorted(df_season['week'].unique())
    total = 0
    
    for week in weeks:
        week_df = df_season[df_season['week'] == week].copy()
        n = len(week_df)
        if n == 0:
            continue
        
        week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False)
        week_df['fan_rank_pct'] = week_df['fan_rank'] / n
        week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False)
        week_df['judge_rank_pct'] = week_df['judge_rank'] / n
        
        total += (week_df['fan_rank_pct'] - week_df['judge_rank_pct']).abs().sum()
    
    return total


# ============================================================================
# 标准化函数
# ============================================================================

def zscore_normalize(values):
    """Z-score标准化"""
    values = np.array(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std == 0 or np.isnan(std):
        return np.zeros_like(values)
    return (values - mean) / std


def minmax_normalize(values):
    """Min-Max归一化到[0,1]"""
    values = np.array(values, dtype=float)
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    if max_val == min_val:
        return np.ones_like(values) * 0.5
    return (values - min_val) / (max_val - min_val)


def zscore_then_minmax(values):
    """先Z-score再Min-Max"""
    z = zscore_normalize(values)
    return minmax_normalize(z)


# ============================================================================
# 主计算流程
# ============================================================================

def compute_all_metrics(df_season, df_trends, schedule, season, lambda_vec, regime='percent'):
    """
    计算给定 λ 向量下的所有原始指标值
    
    关键逻辑：
    - λ 向量影响模拟淘汰结果 → 改变 survival_weeks → 影响 Celebrity, Dancer, Satisfaction
    - λ 向量影响反推的粉丝投票 → 影响 Discrepancy
    - CCVI 基于 Google Trends（不随 λ 变化）
    """
    # 如果传入单一 λ 值，转换为向量
    n_weeks = df_season['week'].nunique()
    if not isinstance(lambda_vec, (list, np.ndarray)):
        lambda_vec = [lambda_vec] * n_weeks
    
    # 模拟淘汰（λ 向量影响谁被淘汰）→ 得到模拟的 survival_weeks
    survival_weeks, simulated_results = simulate_elimination(df_season, lambda_vec, regime)
    
    # 提取历史真实淘汰结果（用于 Discrepancy 反推）
    _, historical_results = extract_historical_elimination(df_season, regime)
    
    # 计算五个原始指标
    # Celebrity, Dancer, Satisfaction 基于模拟的 survival_weeks（随 λ 变化）
    celebrity = compute_celebrity_benefit(df_season, survival_weeks)
    dancer = compute_dancer_effect(df_season, survival_weeks)
    satisfaction = compute_fan_satisfaction(df_season, survival_weeks)
    
    # CCVI 基于 Google Trends（不随 λ 变化）
    ccvi = compute_ccvi(df_trends, schedule, season)
    
    # Discrepancy：基于历史淘汰 + λ 向量反推粉丝投票 → |fan_rank_pct - judge_rank_pct|
    # 分歧度越小 → λ 越合理
    discrepancy = compute_discrepancy_from_history(historical_results, lambda_vec)
    
    return {
        'celebrity': celebrity,
        'dancer': dancer,
        'ccvi': ccvi,
        'satisfaction': satisfaction,
        'discrepancy': discrepancy,
        'survival_weeks': survival_weeks
    }


def warmup_phase(df_main, df_fan, df_trends, schedule):
    """
    预热阶段：用λ=0.5对所有赛季计算原始指标值
    然后对所有指标统一做 Z-score + Min-Max 归一化
    """
    print("=" * 60)
    print("预热阶段：用λ=0.5计算各赛季原始指标")
    print("=" * 60)
    
    all_metrics = []
    
    seasons = sorted(df_fan['season'].unique())
    
    for season in seasons:
        # df_fan 已经在 load_data 时 merge 了人气数据
        df_season = df_fan[df_fan['season'] == season].copy()
        
        # 处理缺失的人气数据
        df_season['Celebrity_Average_Popularity_Score'] = df_season['Celebrity_Average_Popularity_Score'].fillna(0)
        df_season['ballroom_partner_Average_Popularity_Score'] = df_season['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        regime = df_season['regime'].iloc[0] if 'regime' in df_season.columns else 'percent'
        
        # λ=0.5（单一值，不是向量）
        metrics = compute_all_metrics(df_season, df_trends, schedule, season, 0.5, regime)
        
        all_metrics.append({
            'season': season,
            'celebrity': metrics['celebrity'],
            'dancer': metrics['dancer'],
            'ccvi': metrics['ccvi'],
            'satisfaction': metrics['satisfaction'],
            'discrepancy': metrics['discrepancy']
        })
        
        print(f"  Season {season}: celebrity={metrics['celebrity']:.2f}, dancer={metrics['dancer']:.2f}, "
              f"ccvi={metrics['ccvi']:.2f}, satisfaction={metrics['satisfaction']:.3f}, "
              f"discrepancy={metrics['discrepancy']:.3f}")
    
    metrics_df = pd.DataFrame(all_metrics)
    
    print("\n各指标原始值统计：")
    print(metrics_df.describe().round(2))
    
    # 对每个指标做 Z-score + Min-Max
    normalized_df = metrics_df.copy()
    scale_params = {}
    
    for col in ['celebrity', 'dancer', 'ccvi', 'satisfaction', 'discrepancy']:
        raw_values = metrics_df[col].values
        z_values = zscore_normalize(raw_values)
        mm_values = minmax_normalize(z_values)
        normalized_df[f'{col}_norm'] = mm_values
        
        scale_params[col] = {
            'raw_mean': np.nanmean(raw_values),
            'raw_std': np.nanstd(raw_values),
            'z_min': np.nanmin(z_values),
            'z_max': np.nanmax(z_values)
        }
    
    print("\n归一化参数（Z-score → Min-Max）：")
    for col, params in scale_params.items():
        print(f"  {col}: raw_mean={params['raw_mean']:.2f}, raw_std={params['raw_std']:.2f}, "
              f"z_range=[{params['z_min']:.2f}, {params['z_max']:.2f}]")
    
    print("\n归一化后各指标（验证范围[0,1]）：")
    for col in ['celebrity', 'dancer', 'ccvi', 'satisfaction', 'discrepancy']:
        norm_col = f'{col}_norm'
        print(f"  {norm_col}: [{normalized_df[norm_col].min():.3f}, {normalized_df[norm_col].max():.3f}]")
    
    return metrics_df, normalized_df, scale_params


def normalize_metric(value, scale_params, col):
    """
    使用预热阶段的参数对新指标值做标准化
    """
    mean = scale_params[col]['raw_mean']
    std = scale_params[col]['raw_std']
    z_min = scale_params[col]['z_min']
    z_max = scale_params[col]['z_max']
    
    if std == 0:
        return 0.5
    
    z = (value - mean) / std
    if z_max == z_min:
        return 0.5
    return (z - z_min) / (z_max - z_min)


def compute_total_benefit(metrics, scale_params, weights):
    """
    计算归一化后的总收益
    """
    w1, w2, w3, w4, w5 = weights
    
    celeb_norm = normalize_metric(metrics['celebrity'], scale_params, 'celebrity')
    dancer_norm = normalize_metric(metrics['dancer'], scale_params, 'dancer')
    ccvi_norm = normalize_metric(metrics['ccvi'], scale_params, 'ccvi')
    satisfaction_norm = normalize_metric(metrics['satisfaction'], scale_params, 'satisfaction')
    discrepancy_norm = normalize_metric(metrics['discrepancy'], scale_params, 'discrepancy')
    
    # 限制到[0,1]范围
    celeb_norm = np.clip(celeb_norm, 0, 1)
    dancer_norm = np.clip(dancer_norm, 0, 1)
    ccvi_norm = np.clip(ccvi_norm, 0, 1)
    satisfaction_norm = np.clip(satisfaction_norm, 0, 1)
    discrepancy_norm = np.clip(discrepancy_norm, 0, 1)
    
    # 分歧度越低越好，反转
    discrepancy_benefit = 1 - discrepancy_norm
    
    total = (w1 * celeb_norm + 
             w2 * dancer_norm + 
             w3 * ccvi_norm + 
             w4 * satisfaction_norm + 
             w5 * discrepancy_benefit)
    
    return total, {
        'celebrity_norm': celeb_norm,
        'dancer_norm': dancer_norm,
        'ccvi_norm': ccvi_norm,
        'satisfaction_norm': satisfaction_norm,
        'discrepancy_norm': discrepancy_norm,
        'discrepancy_benefit': discrepancy_benefit
    }


def search_optimal_lambda(df_season, df_trends, schedule, season, 
                          scale_params, weights, regime='percent',
                          lambda_options=None):
    """
    网格搜索最佳 λ 向量（每周动态变化）
    搜索空间：每周 λ ∈ {0.2, 0.3, ..., 0.8}
    """
    if lambda_options is None:
        lambda_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 7种选择
    
    n_weeks = df_season['week'].nunique()
    total_combinations = len(lambda_options) ** n_weeks
    
    print(f"    搜索空间: {len(lambda_options)}^{n_weeks} = {total_combinations} 种组合")
    
    best_lambda_vec = [0.5] * n_weeks
    best_benefit = -np.inf
    best_metrics = None
    best_norms = None
    
    # 遍历所有 λ 向量组合
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
        if count % 10000 == 0:
            print(f"      进度: {count}/{total_combinations} ({100*count/total_combinations:.1f}%)")
    
    return best_lambda_vec, best_benefit, best_metrics, best_norms


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("Task 4: 动态权重优化系统 V3")
    print("统一标准化：Z-score → Min-Max → [0,1]")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    df_main, df_fan, df_trends, schedule = load_data()
    print(f"    原始数据集: {len(df_main)} 条记录")
    print(f"    粉丝投票数据: {len(df_fan)} 条记录")
    print(f"    覆盖赛季: {df_fan['season'].min()} - {df_fan['season'].max()}")
    
    # 企业权重
    weights = [0.2, 0.1, 0.3, 0.2, 0.2]  # celebrity, dancer, ccvi, satisfaction, discrepancy
    print(f"\n企业权重: w1(celebrity)={weights[0]}, w2(dancer)={weights[1]}, "
          f"w3(ccvi)={weights[2]}, w4(satisfaction)={weights[3]}, w5(discrepancy)={weights[4]}")
    
    # 预热阶段
    print("\n[2/4] 预热阶段...")
    warmup_raw, warmup_norm, scale_params = warmup_phase(df_main, df_fan, df_trends, schedule)
    
    warmup_raw.to_csv(r'd:\2026-repo\data\task4_warmup_raw_v3.csv', index=False)
    warmup_norm.to_csv(r'd:\2026-repo\data\task4_warmup_norm_v3.csv', index=False)
    
    # 搜索阶段
    print("\n" + "=" * 60)
    print("[3/4] 搜索阶段：寻找各赛季最优 λ 向量（每周动态变化）")
    print("=" * 60)
    
    # λ 范围：0.2 到 0.8，共7种选择
    lambda_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"    λ搜索范围: {lambda_options}")
    
    # 只跑5个代表赛季：3个 percent + 2个 bottom2
    # 选择周数较少的赛季（搜索空间小）
    selected_seasons = [3, 5, 6, 28, 33]  # 3,5,6是percent, 28,33是bottom2
    print(f"    选定赛季: {selected_seasons}")
    
    results = []
    
    for season in selected_seasons:
        # df_fan 已经在 load_data 时 merge 了人气数据
        df_season = df_fan[df_fan['season'] == season].copy()
        
        if len(df_season) == 0:
            print(f"  Season {season}: 无数据，跳过")
            continue
        
        # 处理缺失的人气数据
        df_season['Celebrity_Average_Popularity_Score'] = df_season['Celebrity_Average_Popularity_Score'].fillna(0)
        df_season['ballroom_partner_Average_Popularity_Score'] = df_season['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        regime = df_season['regime'].iloc[0] if 'regime' in df_season.columns else 'percent'
        n_weeks = df_season['week'].nunique()
        
        print(f"\n  Season {season} ({regime}, {n_weeks}周):")
        
        best_lambda_vec, best_benefit, best_metrics, best_norms = search_optimal_lambda(
            df_season, df_trends, schedule, season,
            scale_params, weights, regime, lambda_options
        )
        
        # 基准：λ=[0.5, 0.5, ...]
        baseline_vec = [0.5] * n_weeks
        baseline_metrics = compute_all_metrics(df_season, df_trends, schedule, season, baseline_vec, regime)
        baseline_benefit, _ = compute_total_benefit(baseline_metrics, scale_params, weights)
        
        improvement = best_benefit - baseline_benefit
        
        print(f"    最优 λ 向量: {best_lambda_vec}")
        print(f"    最优收益: {best_benefit:.4f}, 基准收益: {baseline_benefit:.4f}, 改进: {improvement:+.4f}")
        print(f"    Discrepancy: {best_metrics['discrepancy']:.2f} (越小越好)")
        
        results.append({
            'season': season,
            'regime': regime,
            'n_weeks': n_weeks,
            'optimal_lambda_vec': str(best_lambda_vec),
            'optimal_lambda_mean': np.mean(best_lambda_vec),
            'total_benefit': best_benefit,
            'baseline_benefit': baseline_benefit,
            'improvement': improvement,
            'celebrity_raw': best_metrics['celebrity'],
            'dancer_raw': best_metrics['dancer'],
            'ccvi_raw': best_metrics['ccvi'],
            'satisfaction_raw': best_metrics['satisfaction'],
            'discrepancy_raw': best_metrics['discrepancy'],
            'celebrity_norm': best_norms['celebrity_norm'],
            'dancer_norm': best_norms['dancer_norm'],
            'ccvi_norm': best_norms['ccvi_norm'],
            'satisfaction_norm': best_norms['satisfaction_norm'],
            'discrepancy_benefit': best_norms['discrepancy_benefit']
        })
    
    results_df = pd.DataFrame(results)
    
    # 保存结果
    print("\n[4/4] 保存结果...")
    results_df.to_csv(r'd:\2026-repo\data\task4_optimal_results_v3.csv', index=False)
    
    # 统计分析
    print("\n" + "=" * 60)
    print("优化结果统计")
    print("=" * 60)
    
    print(f"\n平均最优λ: {results_df['optimal_lambda_mean'].mean():.3f}")
    print(f"平均总收益: {results_df['total_benefit'].mean():.4f}")
    print(f"平均改进: {results_df['improvement'].mean():.4f}")
    print(f"改进 > 0 的赛季数: {(results_df['improvement'] > 0).sum()} / {len(results_df)}")
    
    print("\n按赛制分组：")
    regime_stats = results_df.groupby('regime').agg({
        'optimal_lambda_mean': 'mean',
        'total_benefit': 'mean',
        'improvement': 'mean'
    }).round(3)
    print(regime_stats)
    
    print("\n完成！结果保存至 data/task4_optimal_results_v3.csv")


if __name__ == "__main__":
    main()
