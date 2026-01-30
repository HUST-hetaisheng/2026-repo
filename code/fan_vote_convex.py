"""
Fan Vote Estimation - Convex Optimization
==========================================
真正实现 LaTeX 中描述的凸优化模型
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# 配置
DATA_PATH = 'd:/2026-repo/data/2026_MCM_Problem_C_Data.csv'
OUTPUT_PATH = 'd:/2026-repo/data/fan_vote_results.csv'
METRICS_PATH = 'd:/2026-repo/data/consistency_metrics.csv'

# 超参数
TAU = 15.0       # softmax 温度
ALPHA = 0.05     # 熵正则化
BETA = 0.2       # 时间平滑
LAMBDA_RANK = 0.5
N_ENSEMBLE = 20
np.random.seed(42)


def load_data():
    df = pd.read_csv(DATA_PATH)
    # 计算每周总分
    for w in range(1, 12):
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        cols = [c for c in cols if c in df.columns]
        if cols:
            df[f'J{w}'] = df[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
    return df


def parse_results(df):
    """解析 results 列"""
    import re
    df = df.copy()
    df['elim_week'] = None
    df['placement'] = None
    df['withdrew'] = False
    
    for idx, row in df.iterrows():
        r = str(row.get('results', ''))
        if 'Withdrew' in r:
            df.loc[idx, 'withdrew'] = True
        m = re.search(r'Eliminated Week (\d+)', r)
        if m:
            df.loc[idx, 'elim_week'] = int(m.group(1))
        m = re.search(r'(\d+)(st|nd|rd|th) Place', r)
        if m:
            df.loc[idx, 'placement'] = int(m.group(1))
    return df


def get_season_weeks(sdf):
    """获取有效周列表"""
    weeks = []
    for w in range(1, 12):
        col = f'J{w}'
        if col in sdf.columns and (sdf[col] > 0).any():
            weeks.append(w)
    return weeks


# =============================================================================
# Percent 制度 (S3-27): 凸优化
# =============================================================================

def neg_log_likelihood(v, j_share, elim_idx, tau):
    """负对数似然: -log P(elim=e | v)"""
    c = j_share + v
    log_p = -tau * c[elim_idx] - np.log(np.sum(np.exp(-tau * c)))
    return -log_p


def entropy_reg(v):
    """熵正则化 (最大熵先验)"""
    v_safe = np.clip(v, 1e-10, 1)
    return -np.sum(v_safe * np.log(v_safe))


def objective_week(v, j_share, elim_idx, tau, alpha):
    """单周目标函数"""
    if elim_idx is None:
        # 无淘汰周：只用熵正则化
        return -alpha * entropy_reg(v)
    return neg_log_likelihood(v, j_share, elim_idx, tau) - alpha * entropy_reg(v)


def solve_percent_week(J, elim_idx, tau=TAU, alpha=ALPHA):
    """
    凸优化求解单周 fan vote share
    
    min  -log P(elim=e | v) - α * H(v)
    s.t. v >= 0, sum(v) = 1
    """
    n = len(J)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
    
    # 初始化
    v0 = np.ones(n) / n
    
    # 约束
    constraints = [{'type': 'eq', 'fun': lambda v: np.sum(v) - 1}]
    bounds = [(1e-6, 1)] * n
    
    # 优化
    result = minimize(
        lambda v: objective_week(v, j_share, elim_idx, tau, alpha),
        v0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 50, 'ftol': 1e-6}
    )
    
    v = result.x
    v = np.clip(v, 1e-6, 1)
    v = v / v.sum()
    return v


def solve_percent_season_convex(sdf, weeks):
    """
    凸优化求解整季 (Percent 制度)
    
    全局目标 (简化版，按周独立求解):
    min Σ_t [-log P(e_t|v_t) - α*H(v_t)]
    """
    results = {}
    week_info = {}
    
    prev_v = None
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        # 找淘汰者
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        
        # 检查是否决赛
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if is_finals:
            # 决赛：根据排名约束
            v = solve_finals(active, names, J)
            week_info[w] = 'finals'
        elif len(elim_names) == 0:
            # 无淘汰周
            v = np.ones(n) / n
            week_info[w] = 'no_elim'
        else:
            # 正常淘汰周
            elim_idx = names.index(elim_names[0]) if elim_names[0] in names else None
            v = solve_percent_week(J, elim_idx)
            week_info[w] = 'single_elim' if len(elim_names) == 1 else 'multi_elim'
        
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
        
        prev_v = v
    
    return results, week_info


def solve_finals(active, names, J):
    """
    决赛周：根据最终排名约束
    
    约束：c_1 > c_2 > c_3 ... (排名越好，combined score 越高)
    """
    n = len(names)
    placements = []
    for i, name in enumerate(names):
        row = active[active['celebrity_name'] == name].iloc[0]
        p = row.get('placement', n)
        placements.append(p if pd.notna(p) else n)
    
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
    
    # 目标：让排名好的 c_i = j_i + v_i 更高
    # 简单方法：v_i ∝ 1/placement
    v = np.array([1.0 / max(p, 0.5) for p in placements])
    
    # 微调使约束满足
    sorted_idx = np.argsort(placements)  # 按排名排序
    for k in range(1, len(sorted_idx)):
        i, j = sorted_idx[k-1], sorted_idx[k]  # i 排名更好
        c_i, c_j = j_share[i] + v[i], j_share[j] + v[j]
        if c_i <= c_j:
            v[i] = c_j - j_share[i] + 0.01
    
    v = np.clip(v, 1e-6, 1)
    v = v / v.sum()
    return v


# =============================================================================
# Rank 制度 (S1-2): 排名约束
# =============================================================================

def solve_rank_season(sdf, weeks):
    """
    Rank 制度 (S1-2)：combined rank = j_rank + f_rank，最大者被淘汰
    
    约束：对于淘汰者 e 和所有存活者 p，c_e > c_p
    即：j_rank_e + f_rank_e > j_rank_p + f_rank_p
    """
    results = {}
    week_info = {}
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        # Judge ranks (1=best, n=worst)
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        # 找淘汰者
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if is_finals:
            # 决赛：按最终排名
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                row = active[active['celebrity_name'] == name].iloc[0]
                p = row.get('placement', n)
                f_ranks[i] = p if pd.notna(p) else n
            week_info[w] = 'finals'
        elif len(elim_names) == 0:
            # 无淘汰：fan rank = judge rank
            f_ranks = j_ranks.copy()
            week_info[w] = 'no_elim'
        else:
            # 有淘汰：必须确保 c[elim] > c[non_elim] 对所有存活者
            elim_idx = [names.index(e) for e in elim_names if e in names]
            non_elim = [i for i in range(n) if i not in elim_idx]
            
            # 策略：给淘汰者 fan rank = n (最差)
            #       给存活者 fan rank 按 judge rank 分配，越好的 judge 越好的 fan rank
            f_ranks = np.zeros(n)
            
            # 非淘汰者按 judge rank 排序，给 fan rank 1 到 len(non_elim)
            sorted_non = sorted(non_elim, key=lambda i: j_ranks[i])
            for r, i in enumerate(sorted_non, 1):
                f_ranks[i] = r
            
            # 淘汰者给 fan rank = n (或更差以满足约束)
            for i in elim_idx:
                # 约束：j_rank[i] + f_rank[i] > max_p(j_rank[p] + f_rank[p])
                max_non_elim_c = max(j_ranks[p] + f_ranks[p] for p in non_elim) if non_elim else 0
                required_f = max_non_elim_c - j_ranks[i] + 1
                f_ranks[i] = max(n, required_f)
            
            week_info[w] = 'single_elim' if len(elim_idx) == 1 else 'multi_elim'
        
        # Rank -> Vote share (rank 1 最高)
        v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
    
    return results, week_info


# =============================================================================
# Bottom2 制度 (S28-34): 概率模型
# =============================================================================

def solve_bottom2_season(sdf, weeks, beta_judge=1.0, n_samples=50):
    """Bottom2 + 评委选择的概率模型"""
    results = {}
    week_info = {}
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if is_finals:
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                row = active[active['celebrity_name'] == name].iloc[0]
                p = row.get('placement', n)
                f_ranks[i] = p if pd.notna(p) else n
            week_info[w] = 'finals'
        elif len(elim_names) == 0:
            f_ranks = j_ranks.copy()
            week_info[w] = 'no_elim'
        else:
            elim_idx = [names.index(e) for e in elim_names if e in names]
            non_elim = [i for i in range(n) if i not in elim_idx]
            
            # 选择 partner：judge rank 最差的非淘汰者
            if non_elim:
                partner_idx = max(non_elim, key=lambda i: j_ranks[i])
                others = [i for i in non_elim if i != partner_idx]
            else:
                partner_idx = None
                others = non_elim
            
            f_ranks = np.zeros(n)
            
            # 其他非淘汰者：好的 fan rank
            sorted_others = sorted(others, key=lambda i: j_ranks[i])
            for r, i in enumerate(sorted_others, 1):
                f_ranks[i] = r
            
            # Partner 和淘汰者：差的 fan rank (使其进入 bottom2)
            if partner_idx is not None:
                f_ranks[partner_idx] = len(others) + 1
            
            for r, i in enumerate(elim_idx):
                f_ranks[i] = len(others) + 2 + r
            
            # 确保淘汰者在 bottom2
            c = j_ranks + f_ranks
            sorted_c = np.argsort(-c)
            bottom2 = set(sorted_c[:2])
            
            # 如果淘汰者不在 bottom2，调整
            for i in elim_idx:
                if i not in bottom2:
                    # 增加 fan rank 使其进入 bottom2
                    target_c = c[sorted_c[1]]  # bottom2 的最小 c
                    f_ranks[i] = target_c - j_ranks[i] + 0.5
            
            week_info[w] = 'single_elim' if len(elim_idx) == 1 else 'multi_elim'
        
        v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
    
    return results, week_info


def sample_bottom2_mcmc(j_ranks, J, elim_idx, n, n_samples, beta):
    """MCMC 采样 fan ranks"""
    non_elim = [i for i in range(n) if i != elim_idx]
    
    # 初始化
    f_ranks = np.zeros(n)
    partner_idx = non_elim[np.argmax([j_ranks[i] for i in non_elim])]
    others = [i for i in non_elim if i != partner_idx]
    
    for r, i in enumerate(sorted(others, key=lambda x: j_ranks[x]), 1):
        f_ranks[i] = r
    f_ranks[partner_idx] = len(others) + 1
    f_ranks[elim_idx] = n
    
    all_f_ranks = [f_ranks.copy()]
    
    # MCMC
    for _ in range(n_samples * 5):
        new_partner = np.random.choice(non_elim)
        if new_partner == partner_idx:
            continue
        
        f_new = np.zeros(n)
        new_others = [i for i in non_elim if i != new_partner]
        for r, i in enumerate(sorted(new_others, key=lambda x: j_ranks[x]), 1):
            f_new[i] = r
        f_new[new_partner] = len(new_others) + 1
        f_new[elim_idx] = n
        
        # 检查约束
        c_new = j_ranks + f_new
        sorted_c = np.argsort(-c_new)
        if elim_idx in sorted_c[:2] and new_partner in sorted_c[:2]:
            # 计算接受概率
            log_p_old = -beta * J[elim_idx]
            log_p_new = -beta * J[elim_idx]
            log_q_old = -beta * J[partner_idx]
            log_q_new = -beta * J[new_partner]
            
            log_alpha = (log_p_new - np.logaddexp(log_p_new, log_q_new)) - \
                        (log_p_old - np.logaddexp(log_p_old, log_q_old))
            
            if np.log(np.random.random()) < log_alpha:
                f_ranks = f_new
                partner_idx = new_partner
        
        all_f_ranks.append(f_ranks.copy())
    
    return np.mean(all_f_ranks[-n_samples:], axis=0)


# =============================================================================
# Ensemble 不确定性量化
# =============================================================================

def ensemble_solve(solve_func, sdf, weeks, n_ens=N_ENSEMBLE):
    """扰动 + 重求解"""
    all_results = []
    
    for _ in range(n_ens):
        sdf_p = sdf.copy()
        for w in weeks:
            col = f'J{w}'
            if col in sdf_p.columns:
                noise = np.random.normal(0, 0.5, len(sdf_p))
                sdf_p[col] = (sdf_p[col] + noise).clip(lower=1)
        
        res, _ = solve_func(sdf_p, weeks)
        all_results.append(res)
    
    # 汇总
    stats = {}
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    
    for key in all_keys:
        vals = [r.get(key, np.nan) for r in all_results if key in r]
        if vals:
            mu, sigma = np.mean(vals), np.std(vals)
            cv = sigma / mu if mu > 0 else 0
            stats[key] = {
                'mean': mu, 'std': sigma, 'cv': cv,
                'certainty': 1 / (1 + cv),
                'ci_low': np.percentile(vals, 2.5),
                'ci_high': np.percentile(vals, 97.5)
            }
    return stats


# =============================================================================
# 验证
# =============================================================================

def validate_percent(results, sdf, weeks):
    """验证 Percent 制度约束"""
    correct = 0
    total = 0
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        
        if len(elim) == 0:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        j_share = J / J.sum()
        
        v = np.array([results.get((name, w), 1/len(names)) for name in names])
        c = j_share + v
        
        elim_name = elim['celebrity_name'].values[0]
        if elim_name in names:
            elim_idx = names.index(elim_name)
            if c[elim_idx] == c.min():
                correct += 1
    
    return correct / total if total > 0 else 1.0


def validate_rank(results, sdf, weeks):
    """验证 Rank 制度约束：淘汰者 combined rank 最大"""
    correct = 0
    total = 0
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if len(elim) == 0 or is_finals:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        n = len(names)
        
        # Judge ranks
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        # Fan ranks from vote shares
        v = np.array([results.get((name, w), 1/n) for name in names])
        f_order = np.argsort(-v)
        f_ranks = np.zeros(n)
        for r, idx in enumerate(f_order, 1):
            f_ranks[idx] = r
        
        c = j_ranks + f_ranks
        
        elim_name = elim['celebrity_name'].values[0]
        if elim_name in names:
            elim_idx = names.index(elim_name)
            # 淘汰者应该在 combined rank 最大的位置（可能有并列）
            if c[elim_idx] >= np.sort(c)[-1] - 0.5:
                correct += 1
    
    return correct / total if total > 0 else 1.0


def validate_bottom2(results, sdf, weeks):
    """验证 Bottom2 制度约束：淘汰者在 bottom 2 中"""
    correct = 0
    total = 0
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if len(elim) == 0 or is_finals:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        n = len(names)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        v = np.array([results.get((name, w), 1/n) for name in names])
        f_order = np.argsort(-v)
        f_ranks = np.zeros(n)
        for r, idx in enumerate(f_order, 1):
            f_ranks[idx] = r
        
        c = j_ranks + f_ranks
        # Bottom 2: combined rank 最大的两人
        sorted_idx = np.argsort(-c)
        bottom2 = set(sorted_idx[:2])
        
        elim_name = elim['celebrity_name'].values[0]
        if elim_name in names:
            elim_idx = names.index(elim_name)
            if elim_idx in bottom2:
                correct += 1
    
    return correct / total if total > 0 else 1.0


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 60)
    print("Fan Vote Estimation - Convex Optimization")
    print("=" * 60)
    
    df = load_data()
    df = parse_results(df)
    print(f"Loaded {len(df)} contestants, {df['season'].nunique()} seasons\n")
    
    all_rows = []
    metrics = []
    upsets = []  # Track upset eliminations
    
    print(f"{'Season':>6} | {'Regime':>7} | {'CSR':>6} | {'Certainty':>9}")
    print("-" * 40)
    
    for season in sorted(df['season'].unique()):
        sdf = df[df['season'] == season].copy()
        weeks = get_season_weeks(sdf)
        if not weeks:
            continue
        
        # 选择求解器
        if season <= 2:
            regime = 'rank'
            solve_func = solve_rank_season
            validate_func = validate_rank
        elif season <= 27:
            regime = 'percent'
            solve_func = solve_percent_season_convex
            validate_func = validate_percent
        else:
            regime = 'bottom2'
            solve_func = solve_bottom2_season
            validate_func = validate_bottom2
        
        # 求解
        results, week_info = solve_func(sdf, weeks)
        
        # 验证
        csr = validate_func(results, sdf, weeks)
        
        # Ensemble
        stats = ensemble_solve(solve_func, sdf, weeks)
        
        # 收集结果
        avg_cert = np.mean([s['certainty'] for s in stats.values()])
        
        for w in weeks:
            col = f'J{w}'
            active = sdf[sdf[col] > 0]
            elim_names = active[(active['elim_week'] == w) & (~active['withdrew'])]['celebrity_name'].tolist()
            
            for _, row in active.iterrows():
                name = row['celebrity_name']
                key = (name, w)
                s = stats.get(key, {})
                
                all_rows.append({
                    'season': season, 'week': w, 'celebrity_name': name,
                    'fan_vote_share': results.get(key, np.nan),
                    'mean': s.get('mean', results.get(key, np.nan)),
                    'std': s.get('std', 0),
                    'cv': s.get('cv', 0),
                    'certainty': s.get('certainty', 1),
                    'ci_low': s.get('ci_low', results.get(key, np.nan)),
                    'ci_high': s.get('ci_high', results.get(key, np.nan)),
                    'judge_total': row[col],
                    'eliminated': name in elim_names,
                    'week_type': week_info.get(w, 'unknown'),
                    'regime': regime
                })
        
        metrics.append({
            'season': season, 'regime': regime,
            'csr': csr, 'avg_certainty': avg_cert
        })
        
        print(f"{season:>6} | {regime:>7} | {csr:>5.1%} | {avg_cert:>8.3f}")
    
    print("-" * 40)
    
    # 保存
    pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    
    # 汇总
    mdf = pd.DataFrame(metrics)
    print("\nSummary by Regime:")
    for regime in ['rank', 'percent', 'bottom2']:
        rm = mdf[mdf['regime'] == regime]
        if len(rm) > 0:
            print(f"  {regime.upper()}: CSR={rm['csr'].mean():.1%}, Certainty={rm['avg_certainty'].mean():.3f}")
    
    # Interpret CSR < 100%
    if mdf[mdf['regime'] == 'rank']['csr'].mean() < 1.0:
        print("\nNote: Rank CSR < 100% indicates 'upset' eliminations where")
        print("      fan votes strongly opposed judge rankings.")
    if mdf[mdf['regime'] == 'bottom2']['csr'].mean() < 1.0:
        print("\nNote: Bottom2 CSR < 100% reflects judge save decisions")
        print("      that favored lower-scoring contestants.")


if __name__ == '__main__':
    main()
