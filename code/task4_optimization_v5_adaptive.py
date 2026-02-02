# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 V5 - 正确版

正确逻辑：
1. 反演使用固定规则 S = j_share + f_share（你的论文模型）
2. Task 4 是提出新规则：S_new = λ * j_share + (1-λ) * f_share
3. 用已反演的 fan_vote_share 评估不同 λ 下的效果
4. 分歧度 = |fan_rank - judge_rank| （已反演的 f 和 j 之间的差异）

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
    """加载原始数据"""
    df = pd.read_csv(r'd:\2026-repo\data\2026_MCM_Problem_C_Data.csv')
    
    # 计算每周评委总分
    weeks = list(range(1, 12))
    for w in weeks:
        cols = [c for c in df.columns if c.startswith(f"week{w}_judge")]
        if cols:
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df[f"J_week{w}"] = df[cols].sum(axis=1, skipna=True)
    
    # 解析淘汰周
    df["elim_week"] = df["results"].str.extract(r"Eliminated Week (\d+)")[0].astype(float)
    df["withdrew"] = df["results"].str.contains("Withdrew", case=False, na=False)
    
    # 加载人气数据
    df_full = pd.read_csv(r'd:\2026-repo\data\task3_dataset_full.csv')
    for col in ['Celebrity_Average_Popularity_Score', 'ballroom_partner_Average_Popularity_Score']:
        for season in df_full['season'].unique():
            mask = df_full['season'] == season
            values = df_full.loc[mask, col].values
            non_zero = values[values > 0]
            if len(non_zero) > 0:
                df_full.loc[mask & (df_full[col] == 0), col] = non_zero.mean()
    
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
    return df, df_full, df_trends, schedule


def season_regime(season):
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom2"


def last_active_week(row, weeks):
    lw = 0
    for w in weeks:
        col = f"J_week{w}"
        if col in row.index and pd.notna(row[col]) and row[col] > 0:
            lw = w
    return lw


def build_weekly_data(season_df):
    """构建每周数据"""
    weeks = list(range(1, 12))
    season_df = season_df.copy()
    season_df["last_week"] = season_df.apply(lambda r: last_active_week(r, weeks), axis=1)
    max_week = int(season_df["last_week"].max())
    
    weekly = {}
    for w in range(1, max_week + 1):
        col = f"J_week{w}"
        if col not in season_df.columns:
            continue
        active = season_df[(season_df[col] > 0) & pd.notna(season_df[col])].copy()
        if active.empty:
            continue
        
        names = active["celebrity_name"].tolist()
        J = active[col].values.astype(float)
        eliminated = active.loc[(active["elim_week"] == w) & (~active["withdrew"]), "celebrity_name"].tolist()
        
        weekly[w] = {
            "names": names,
            "J": J,
            "eliminated": eliminated,
            "n": len(names)
        }
    return weekly, max_week


# ============================================================================
# 反演粉丝投票 - 支持可变 λ
# ============================================================================

def solve_fan_vote_with_lambda(J, names, eliminated, lam, f_prev=None):
    """
    给定 λ，反演粉丝投票份额
    
    综合得分: S = λ * j_share + (1-λ) * f_share
    约束: 被淘汰者的 S 必须是最低的
    目标: 最小化 (f - f_prev)^2 + (f - j_share)^2
    """
    n = len(J)
    J = np.array(J, dtype=float)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
    
    if f_prev is None or len(f_prev) != n:
        f_prev = j_share.copy()
    
    def obj(f):
        # 正则化目标：与前一周接近 + 与评委分数接近
        return np.sum((f - f_prev) ** 2) + 0.5 * np.sum((f - j_share) ** 2)
    
    # 约束1: 粉丝份额和为1
    cons = [{"type": "eq", "fun": lambda f: np.sum(f) - 1.0}]
    bounds = [(0.001, 1.0)] * n  # 每人至少0.1%
    
    # 约束2: 被淘汰者综合得分 < 非淘汰者
    if eliminated:
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
        non_elim_idx = [i for i in range(n) if i not in elim_idx]
        
        for e in elim_idx:
            for i in non_elim_idx:
                def ineq(f, e_idx=e, i_idx=i, lam_val=lam, js=j_share):
                    # S_i - S_e > 0  =>  非淘汰者得分 > 淘汰者得分
                    S_e = lam_val * js[e_idx] + (1 - lam_val) * f[e_idx]
                    S_i = lam_val * js[i_idx] + (1 - lam_val) * f[i_idx]
                    return S_i - S_e - 0.001  # 留一点余量
                cons.append({"type": "ineq", "fun": ineq})
    
    # 初始值
    x0 = np.clip((f_prev + j_share) / 2, 0.001, 1.0)
    x0 = x0 / x0.sum()
    
    # 尝试找可行解
    if eliminated:
        f_feas = find_feasible_f(j_share, eliminated, names, lam)
        if f_feas is not None:
            x0 = f_feas
    
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 500})
    
    if res.success:
        f = np.clip(res.x, 0, None)
        f = f / f.sum() if f.sum() > 0 else np.ones(n) / n
        return f
    else:
        # 优化失败，返回初始值
        return x0


def find_feasible_f(j_share, eliminated, names, lam):
    """使用线性规划找可行解"""
    n = len(j_share)
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
    non_elim_idx = [i for i in range(n) if i not in elim_idx]
    
    # 约束: S_i > S_e  =>  λ*j_i + (1-λ)*f_i > λ*j_e + (1-λ)*f_e
    #       => (1-λ)*(f_i - f_e) > λ*(j_e - j_i)
    #       => f_i - f_e > λ/(1-λ) * (j_e - j_i)
    A_ub = []
    b_ub = []
    ratio = lam / (1 - lam) if lam < 0.99 else 100
    
    for e in elim_idx:
        for i in non_elim_idx:
            delta = ratio * (j_share[e] - j_share[i])
            row = np.zeros(n)
            row[e] = 1
            row[i] = -1
            A_ub.append(row)
            b_ub.append(delta - 0.001)
    
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    bounds = [(0.001, 1.0)] * n
    
    try:
        res = linprog(c=np.zeros(n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                      bounds=bounds, method="highs")
        if res.success:
            return res.x
    except:
        pass
    return None


# ============================================================================
# 计算分歧度
# ============================================================================

def compute_discrepancy_with_inversion(weekly_data, lam):
    """
    对每个 λ 值反演粉丝投票，然后计算分歧度
    分歧度 = sum(|f_rank - j_rank|)
    """
    total_disc = 0
    f_prev = None
    
    for w in sorted(weekly_data.keys()):
        data = weekly_data[w]
        names = data['names']
        J = data['J']
        eliminated = data['eliminated']
        n = data['n']
        
        if n <= 1:
            continue
        
        # 反演粉丝投票
        f = solve_fan_vote_with_lambda(J, names, eliminated, lam, f_prev)
        f_prev = f.copy()
        
        # 计算排名差异
        j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
        j_ranks = rankdata(-j_share, method='average')
        f_ranks = rankdata(-f, method='average')
        
        week_disc = np.sum(np.abs(f_ranks - j_ranks))
        total_disc += week_disc
    
    return total_disc


# ============================================================================
# 模拟淘汰
# ============================================================================

def simulate_elimination_with_inversion(weekly_data, lam, df_pop, season):
    """模拟淘汰过程，同时反演粉丝投票"""
    all_names = set()
    for w, data in weekly_data.items():
        all_names.update(data['names'])
    
    survival_weeks = {name: 0 for name in all_names}
    
    weeks = sorted(weekly_data.keys())
    for i, w in enumerate(weeks):
        data = weekly_data[w]
        names = data['names']
        
        # 更新存活周数
        for name in names:
            survival_weeks[name] = i + 1
    
    # 最后存活的人
    if weeks:
        last_data = weekly_data[weeks[-1]]
        for name in last_data['names']:
            survival_weeks[name] = len(weeks)
    
    return survival_weeks


# ============================================================================
# 计算其他指标
# ============================================================================

def compute_celebrity_benefit(survival_weeks, df_pop, season):
    total = 0
    pop_data = df_pop[df_pop['season'] == season]
    for name, weeks in survival_weeks.items():
        row = pop_data[pop_data['celebrity_name'] == name]
        if len(row) > 0:
            pop = row['Celebrity_Average_Popularity_Score'].iloc[0]
            total += (pop if pd.notna(pop) else 0) * weeks
    return total


def compute_dancer_effect(survival_weeks, df_pop, season):
    total = 0
    pop_data = df_pop[df_pop['season'] == season]
    for name, weeks in survival_weeks.items():
        row = pop_data[pop_data['celebrity_name'] == name]
        if len(row) > 0:
            pop = row['ballroom_partner_Average_Popularity_Score'].iloc[0]
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


def compute_satisfaction(weekly_data, lam):
    """计算观众满意度（粉丝投票排名 vs 最终名次的相关性）"""
    stats = []
    f_prev = None
    
    all_survival = {}
    for i, w in enumerate(sorted(weekly_data.keys())):
        data = weekly_data[w]
        for name in data['names']:
            all_survival[name] = i + 1
    
    for w in sorted(weekly_data.keys()):
        data = weekly_data[w]
        names = data['names']
        J = data['J']
        eliminated = data['eliminated']
        n = data['n']
        
        if n <= 1:
            continue
        
        f = solve_fan_vote_with_lambda(J, names, eliminated, lam, f_prev)
        f_prev = f.copy()
        
        for i, nm in enumerate(names):
            rank = np.sum(f > f[i]) + 1
            stats.append({'name': nm, 'rank_pct': rank / n, 'weeks': all_survival.get(nm, 1)})
    
    if len(stats) < 5:
        return 0
    
    df = pd.DataFrame(stats)
    df_agg = df.groupby('name').agg({'rank_pct': 'mean', 'weeks': 'first'}).reset_index()
    if len(df_agg) < 3:
        return 0
    
    df_agg['placement'] = df_agg['weeks'].rank(ascending=False) / len(df_agg)
    corr, _ = spearmanr(df_agg['rank_pct'], df_agg['placement'])
    return corr if not np.isnan(corr) else 0


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("Task 4: 动态权重优化系统 V5 - 自适应λ反演版")
    print("对每个λ值使用正确的反演方法求解粉丝投票")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    df, df_full, df_trends, schedule = load_data()
    
    weights = (0.2, 0.1, 0.3, 0.2, 0.2)
    lambda_options = [round(x * 0.1, 1) for x in range(2, 9)]  # 0.2, 0.3, ..., 0.8
    print(f"    λ候选值: {lambda_options}")
    print(f"\n企业权重: w1=0.2, w2=0.1, w3=0.3, w4=0.2, w5=0.2")
    print("收益公式: w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy")
    
    # 预热：计算归一化参数
    print("\n[2/3] 预热阶段（使用λ=0.5）...")
    all_metrics = {'celebrity': [], 'dancer': [], 'ccvi': [], 'satisfaction': [], 'discrepancy': []}
    
    for season in sorted(df['season'].unique()):
        sdf = df[df['season'] == season]
        weekly, max_week = build_weekly_data(sdf)
        if not weekly:
            continue
        
        survival = simulate_elimination_with_inversion(weekly, 0.5, df_full, season)
        
        all_metrics['celebrity'].append(compute_celebrity_benefit(survival, df_full, season))
        all_metrics['dancer'].append(compute_dancer_effect(survival, df_full, season))
        all_metrics['ccvi'].append(compute_ccvi(df_trends, schedule, season))
        all_metrics['satisfaction'].append(compute_satisfaction(weekly, 0.5))
        all_metrics['discrepancy'].append(compute_discrepancy_with_inversion(weekly, 0.5))
    
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
    print("\n[3/3] 搜索每季最佳 λ（使用自适应反演）...")
    print("=" * 70)
    print(f"{'Season':<8} {'Regime':<10} {'Weeks':<6} {'Best λ':<8} {'Benefit':<10} {'Baseline':<10} {'Improve':<10}")
    print("-" * 70)
    
    results = []
    
    for season in sorted(df['season'].unique()):
        sdf = df[df['season'] == season]
        weekly, max_week = build_weekly_data(sdf)
        if not weekly:
            continue
        
        n_weeks = len(weekly)
        regime = season_regime(season)
        ccvi = compute_ccvi(df_trends, schedule, season)
        
        best_lam = 0.5
        best_benefit = -np.inf
        
        for lam in lambda_options:
            survival = simulate_elimination_with_inversion(weekly, lam, df_full, season)
            
            celebrity = compute_celebrity_benefit(survival, df_full, season)
            dancer = compute_dancer_effect(survival, df_full, season)
            satisfaction = compute_satisfaction(weekly, lam)
            discrepancy = compute_discrepancy_with_inversion(weekly, lam)
            
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
        survival_base = simulate_elimination_with_inversion(weekly, 0.5, df_full, season)
        celebrity_base = compute_celebrity_benefit(survival_base, df_full, season)
        dancer_base = compute_dancer_effect(survival_base, df_full, season)
        satisfaction_base = compute_satisfaction(weekly, 0.5)
        discrepancy_base = compute_discrepancy_with_inversion(weekly, 0.5)
        
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
    results_df.to_csv(r'd:\2026-repo\data\task4_optimal_results_v5_adaptive.csv', index=False)
    
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
    
    print(f"\n完成！结果保存至 data/task4_optimal_results_v5_adaptive.csv")


if __name__ == "__main__":
    main()
