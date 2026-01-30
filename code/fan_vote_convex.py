"""
Fan Vote Estimation - Convex Optimization
==========================================
Fixed to address data/label handling, rank permutation validity,
bottom2 inference, hard constraints for percent regime, and
smoother handling of no-elimination weeks.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import warnings

warnings.filterwarnings('ignore')

# Paths (relative to repo root)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / '2026_MCM_Problem_C_Data.csv'
OUTPUT_PATH = BASE_DIR / 'data' / 'fan_vote_results.csv'
METRICS_PATH = BASE_DIR / 'data' / 'consistency_metrics.csv'

# Hyperparameters
TAU = 15.0       # softmax temperature (kept for optional tie-breaks)
ALPHA = 0.05     # entropy regularization weight
BETA_LIKE = 1.0  # likelihood weight in stage-2
LAMBDA_RANK = 0.5
N_ENSEMBLE = int(os.getenv('N_ENSEMBLE', '20'))
FLOAT_TOL = 1e-6
np.random.seed(42)


def load_data():
    df = pd.read_csv(DATA_PATH)
    # compute weekly judge totals
    for w in range(1, 12):
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        cols = [c for c in cols if c in df.columns]
        if cols:
            df[f'J{w}'] = df[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
    return df


def infer_elim_week_from_scores(row):
    last_week = None
    for w in range(1, 12):
        col = f'J{w}'
        if col in row and pd.notna(row[col]) and row[col] > 0:
            last_week = w
    return last_week


def parse_results(df):
    """Parse results text without overwriting existing placement/elim_week."""
    import re
    df = df.copy()
    if 'elim_week' not in df.columns:
        df['elim_week'] = np.nan
    if 'placement' not in df.columns:
        df['placement'] = np.nan
    if 'withdrew' not in df.columns:
        df['withdrew'] = False

    for idx, row in df.iterrows():
        r = str(row.get('results', ''))
        if 'Withdrew' in r:
            df.loc[idx, 'withdrew'] = True
        if pd.isna(df.loc[idx, 'elim_week']):
            m = re.search(r'Eliminated Week (\d+)', r)
            if m:
                df.loc[idx, 'elim_week'] = int(m.group(1))
        if pd.isna(df.loc[idx, 'placement']):
            m = re.search(r'(\d+)(st|nd|rd|th) Place', r)
            if m:
                df.loc[idx, 'placement'] = int(m.group(1))

    # cross-check elimination week using last positive judge week
    df['elim_week_infer'] = df.apply(infer_elim_week_from_scores, axis=1)
    df['elim_week_final'] = df['elim_week']
    missing = df['elim_week_final'].isna()
    df.loc[missing, 'elim_week_final'] = df.loc[missing, 'elim_week_infer']
    df['elim_week_mismatch'] = (
        df['elim_week'].notna()
        & df['elim_week_infer'].notna()
        & (df['elim_week'] != df['elim_week_infer'])
    )
    return df


def get_season_weeks(sdf):
    weeks = []
    for w in range(1, 12):
        col = f'J{w}'
        if col in sdf.columns and (sdf[col] > 0).any():
            weeks.append(w)
    return weeks


# =============================================================================
# Helpers
# =============================================================================

def entropy_reg(v):
    v_safe = np.clip(v, 1e-10, 1)
    return -np.sum(v_safe * np.log(v_safe))


def fill_no_elim_weeks(week_data, weeks):
    """Fill no-elimination weeks by interpolation of adjacent weeks."""
    for idx, w in enumerate(weeks):
        if week_data[w]['v'] is not None:
            continue
        names = week_data[w]['names']

        prev_w = None
        for wp in reversed(weeks[:idx]):
            if week_data[wp]['v'] is not None and week_data[wp]['names'] == names:
                prev_w = wp
                break
        next_w = None
        for wn in weeks[idx + 1:]:
            if week_data[wn]['v'] is not None and week_data[wn]['names'] == names:
                next_w = wn
                break

        if prev_w is not None and next_w is not None:
            v = (week_data[prev_w]['v'] + week_data[next_w]['v']) / 2.0
        elif prev_w is not None:
            v = week_data[prev_w]['v'].copy()
        elif next_w is not None:
            v = week_data[next_w]['v'].copy()
        else:
            v = np.ones(len(names)) / len(names)

        v = np.clip(v, 1e-8, 1)
        v = v / v.sum()
        week_data[w]['v'] = v
        week_data[w]['week_type'] = week_data[w]['week_type'] + '_interp'

    return week_data


# =============================================================================
# Percent Regime (S3-27): hard constraints + slack
# =============================================================================

def neg_log_likelihood_multi(v, j_share, elim_idxs, tau):
    """Softmax negative log-likelihood (multi-elim tie-break)."""
    if not elim_idxs:
        return 0.0
    c = j_share + v
    k = len(elim_idxs)
    # sum of class log-probs, using shared logsumexp for stability
    return (tau * np.sum(c[elim_idxs])) + k * logsumexp(-tau * c)


def solve_percent_week_hard(J, elim_idxs):
    """
    Solve fan vote shares with hard elimination constraints (slack minimized).

    Constraints: for all e in elim, p in non-elim:
        (j_e + v_e) - (j_p + v_p) <= s
    Minimize: s - alpha * H(v)
    """
    n = len(J)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n

    if not elim_idxs:
        return np.ones(n) / n, 0.0

    non_elim = [i for i in range(n) if i not in elim_idxs]

    def objective(x):
        return x[-1]

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n]) - 1}]
    for e in elim_idxs:
        for p in non_elim:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, e=e, p=p: x[-1] - ((j_share[e] + x[e]) - (j_share[p] + x[p]))
            })

    bounds = [(1e-8, 1.0)] * n + [(0.0, None)]

    v0 = np.ones(n) / n
    if non_elim:
        s0 = max(0.0, max((j_share[e] + v0[e]) - (j_share[p] + v0[p])
                          for e in elim_idxs for p in non_elim))
    else:
        s0 = 0.0
    x0 = np.concatenate([v0, [s0]])

    result = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-9}
    )

    if not result.success:
        v = v0.copy()
        for e in elim_idxs:
            v[e] = 1e-4
        v = v / v.sum()
        slack = 0.0
        if non_elim:
            slack = max(0.0, max((j_share[e] + v[e]) - (j_share[p] + v[p])
                                 for e in elim_idxs for p in non_elim))
        return v, slack

    x = result.x
    v = np.clip(x[:n], 1e-8, 1.0)
    v = v / v.sum()
    slack = max(0.0, x[-1])
    return v, slack


def solve_percent_week_two_stage(J, elim_idxs, tau=TAU, alpha=ALPHA, beta=BETA_LIKE):
    """
    Two-stage solve:
      1) minimize slack s (hard consistency priority)
      2) within s <= s* + eps, minimize beta*negloglike - alpha*H(v)
    """
    n = len(J)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n

    if not elim_idxs:
        return np.ones(n) / n, 0.0

    # Stage 1: minimal slack
    v_stage1, s_star = solve_percent_week_hard(J, elim_idxs)
    eps = max(1e-6, 1e-3 * s_star)

    non_elim = [i for i in range(n) if i not in elim_idxs]

    def objective(v):
        return beta * neg_log_likelihood_multi(v, j_share, elim_idxs, tau) - alpha * entropy_reg(v)

    constraints = [{'type': 'eq', 'fun': lambda v: np.sum(v) - 1}]
    for e in elim_idxs:
        for p in non_elim:
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, e=e, p=p: (s_star + eps) - ((j_share[e] + v[e]) - (j_share[p] + v[p]))
            })

    bounds = [(1e-8, 1.0)] * n

    result = minimize(
        objective, v_stage1, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-9}
    )

    if not result.success:
        return v_stage1, s_star

    v = np.clip(result.x, 1e-8, 1.0)
    v = v / v.sum()
    return v, s_star


def solve_finals_robust(active, names, J):
    """Ensure ordering constraints survive normalization (iterative adjust)."""
    n = len(names)
    placements = []
    for name in names:
        row = active[active['celebrity_name'] == name].iloc[0]
        p = row.get('placement', n)
        placements.append(p if pd.notna(p) else n)

    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
    sorted_idx = np.argsort(placements)

    # initialize once, then iteratively adjust without reset
    v = np.array([1.0 / max(p, 0.5) for p in placements])
    v = np.clip(v, 1e-6, 1)
    v = v / v.sum()

    for _ in range(30):
        for k in range(1, len(sorted_idx)):
            i, j = sorted_idx[k - 1], sorted_idx[k]
            c_i, c_j = j_share[i] + v[i], j_share[j] + v[j]
            if c_i <= c_j + FLOAT_TOL:
                v[i] += (c_j - c_i) + 0.02 * (k + 1)
        v = np.clip(v, 1e-6, 1)
        v = v / v.sum()

        c = j_share + v
        valid = True
        for k in range(1, len(sorted_idx)):
            i, j = sorted_idx[k - 1], sorted_idx[k]
            if c[i] <= c[j] + FLOAT_TOL:
                valid = False
                break
        if valid:
            return v

    # fallback: strict decreasing weights
    v = np.zeros(n)
    base = 1.0
    for idx in sorted_idx:
        v[idx] = base
        base *= 0.8
    v = v / v.sum()
    return v


def solve_percent_season_convex(sdf, weeks):
    """Percent regime with hard constraints + no-elim interpolation."""
    week_info = {}
    week_data = {}

    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        if len(active) == 0:
            continue

        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)

        elim = active[(active['elim_week_final'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        elim_idxs = [names.index(e) for e in elim_names if e in names]

        is_finals = (w == max(weeks)) and (active['placement'].notna().any())

        if is_finals:
            v = solve_finals_robust(active, names, J)
            week_info[w] = 'finals'
        elif len(elim_idxs) == 0:
            v = None
            week_info[w] = 'no_elim'
        else:
            v, _slack = solve_percent_week_two_stage(J, elim_idxs)
            week_info[w] = 'single_elim' if len(elim_idxs) == 1 else f'multi_elim_{len(elim_idxs)}'

        week_data[w] = {'names': names, 'v': v, 'week_type': week_info[w]}

    week_data = fill_no_elim_weeks(week_data, weeks)

    results = {}
    for w in weeks:
        if w not in week_data:
            continue
        names = week_data[w]['names']
        v = week_data[w]['v']
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
        week_info[w] = week_data[w]['week_type']

    return results, week_info


# =============================================================================
# Rank Regime (S1-2): valid permutations
# =============================================================================

def build_rank_fan_permutation(j_ranks, elim_idxs):
    n = len(j_ranks)
    k = len(elim_idxs)
    non_elim = [i for i in range(n) if i not in elim_idxs]
    f_ranks = np.zeros(n)

    # non-elim: give best fan ranks to worst judges
    sorted_non = sorted(non_elim, key=lambda i: j_ranks[i], reverse=True)
    for r, i in enumerate(sorted_non, 1):
        f_ranks[i] = r

    # elim: give worst fan ranks to best judges
    sorted_elim = sorted(elim_idxs, key=lambda i: j_ranks[i])
    for offset, i in enumerate(sorted_elim):
        f_ranks[i] = n - offset

    def top_k_indices(c, k):
        return set(np.argsort(-c)[:k])

    c = j_ranks + f_ranks
    max_iter = n * n
    for _ in range(max_iter):
        top_k = top_k_indices(c, k)
        if all(i in top_k for i in elim_idxs):
            return f_ranks, True
        bad_elim = [i for i in elim_idxs if i not in top_k]
        bad_non = [i for i in top_k if i not in elim_idxs]
        if not bad_elim or not bad_non:
            break
        best = None
        for e in bad_elim:
            for p in bad_non:
                new_c_e = j_ranks[e] + f_ranks[p]
                new_c_p = j_ranks[p] + f_ranks[e]
                improvement = (new_c_e - c[e]) + (c[p] - new_c_p)
                if best is None or improvement > best[0]:
                    best = (improvement, e, p)
        if best is None or best[0] <= 0:
            break
        _, e, p = best
        f_ranks[e], f_ranks[p] = f_ranks[p], f_ranks[e]
        c = j_ranks + f_ranks

    return f_ranks, False


def solve_rank_season(sdf, weeks):
    results = {}
    week_info = {}
    week_data = {}

    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        if len(active) == 0:
            continue

        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        n = len(active)

        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r

        elim = active[(active['elim_week_final'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        elim_idxs = [names.index(e) for e in elim_names if e in names]
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())

        if is_finals:
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                row = active[active['celebrity_name'] == name].iloc[0]
                p = row.get('placement', n)
                f_ranks[i] = p if pd.notna(p) else n
            v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
            v = v / v.sum()
            week_info[w] = 'finals'
        elif len(elim_idxs) == 0:
            v = None
            week_info[w] = 'no_elim'
        else:
            f_ranks, feasible = build_rank_fan_permutation(j_ranks, elim_idxs)
            v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
            v = v / v.sum()
            if feasible:
                week_info[w] = 'single_elim' if len(elim_idxs) == 1 else f'multi_elim_{len(elim_idxs)}'
            else:
                week_info[w] = 'upset'

        week_data[w] = {'names': names, 'v': v, 'week_type': week_info[w]}

    week_data = fill_no_elim_weeks(week_data, weeks)

    for w in weeks:
        if w not in week_data:
            continue
        names = week_data[w]['names']
        v = week_data[w]['v']
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
        week_info[w] = week_data[w]['week_type']

    return results, week_info


# =============================================================================
# Bottom2 Regime (S28-34): enumerate partner
# =============================================================================

def assign_bottom2_fan_ranks(j_ranks, elim_idx, partner_idx):
    n = len(j_ranks)
    f_ranks = np.zeros(n)
    # give worst fan ranks to bottom2 pair
    if j_ranks[elim_idx] <= j_ranks[partner_idx]:
        worst_order = [elim_idx, partner_idx]
    else:
        worst_order = [partner_idx, elim_idx]
    f_ranks[worst_order[0]] = n
    f_ranks[worst_order[1]] = n - 1

    remaining = [i for i in range(n) if i not in (elim_idx, partner_idx)]
    remaining_ranks = list(range(1, n - 1))
    remaining_sorted = sorted(remaining, key=lambda i: j_ranks[i])
    for r, i in zip(remaining_ranks, remaining_sorted):
        f_ranks[i] = r

    return f_ranks


def solve_bottom2_season(sdf, weeks):
    results = {}
    week_info = {}
    week_data = {}

    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        if len(active) == 0:
            continue

        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        n = len(active)

        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r

        elim = active[(active['elim_week_final'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        elim_idxs = [names.index(e) for e in elim_names if e in names]
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())

        if is_finals:
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                row = active[active['celebrity_name'] == name].iloc[0]
                p = row.get('placement', n)
                f_ranks[i] = p if pd.notna(p) else n
            v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
            v = v / v.sum()
            week_info[w] = 'finals'
        elif len(elim_idxs) == 0:
            v = None
            week_info[w] = 'no_elim'
        elif len(elim_idxs) >= 2:
            # multi-elim: fall back to rank-style permutation
            f_ranks, _feasible = build_rank_fan_permutation(j_ranks, elim_idxs)
            v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
            v = v / v.sum()
            week_info[w] = f'multi_elim_{len(elim_idxs)}'
        else:
            elim_idx = elim_idxs[0]
            non_elim = [i for i in range(n) if i != elim_idx]

            best = None
            feasible_found = False
            for b in non_elim:
                f_ranks = assign_bottom2_fan_ranks(j_ranks, elim_idx, b)
                c = j_ranks + f_ranks
                bottom2 = set(np.argsort(-c)[:2])
                feasible = elim_idx in bottom2 and b in bottom2
                cost = np.sum(np.abs(f_ranks - j_ranks))
                score = (1 if feasible else 0, -cost)
                if best is None or score > best[0]:
                    best = (score, f_ranks, feasible)
                if feasible:
                    feasible_found = True

            f_ranks = best[1]
            v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
            v = v / v.sum()
            if feasible_found and best[2]:
                week_info[w] = 'single_elim'
            else:
                week_info[w] = 'bottom2_relaxed'

        week_data[w] = {'names': names, 'v': v, 'week_type': week_info[w]}

    week_data = fill_no_elim_weeks(week_data, weeks)

    for w in weeks:
        if w not in week_data:
            continue
        names = week_data[w]['names']
        v = week_data[w]['v']
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
        week_info[w] = week_data[w]['week_type']

    return results, week_info


# =============================================================================
# Ensemble
# =============================================================================

def ensemble_solve(solve_func, sdf, weeks, n_ens=N_ENSEMBLE):
    """Perturb scores without reviving eliminated contestants."""
    all_results = []

    for _ in range(n_ens):
        sdf_p = sdf.copy()
        for w in weeks:
            col = f'J{w}'
            if col in sdf_p.columns:
                mask = sdf_p[col] > 0
                noise = np.random.normal(0, 0.5, len(sdf_p))
                sdf_p.loc[mask, col] = (sdf_p.loc[mask, col] + noise[mask]).clip(lower=1)

        res, _ = solve_func(sdf_p, weeks)
        all_results.append(res)

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
# Validation
# =============================================================================

def validate_percent(results, sdf, weeks):
    """Percent regime: all eliminated in bottom-k combined scores."""
    correct = 0
    total = 0

    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week_final'] == w) & (~active['withdrew'])]

        if len(elim) == 0:
            continue

        total += 1
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        j_share = J / J.sum() if J.sum() > 0 else np.ones(len(names)) / len(names)
        v = np.array([results.get((name, w), 1/len(names)) for name in names])
        c = j_share + v

        elim_names = elim['celebrity_name'].tolist()
        k = len(elim_names)
        sorted_idx = np.argsort(c)
        bottom_k = set(sorted_idx[:k])
        all_in_bottom = all(
            names.index(e) in bottom_k
            for e in elim_names if e in names
        )
        if all_in_bottom:
            correct += 1

    return correct / total if total > 0 else 1.0


def validate_rank(results, sdf, weeks):
    """Rank regime: eliminated are in top-k (worst) combined ranks."""
    correct = 0
    total = 0

    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week_final'] == w) & (~active['withdrew'])]
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
        elim_names = elim['celebrity_name'].tolist()
        elim_idxs = [names.index(e) for e in elim_names if e in names]
        k = len(elim_idxs)
        top_k = set(np.argsort(-c)[:k])
        if all(i in top_k for i in elim_idxs):
            correct += 1

    return correct / total if total > 0 else 1.0


def validate_bottom2(results, sdf, weeks):
    """Bottom2 regime: eliminated must be in bottom2 (single elim)."""
    correct = 0
    total = 0

    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week_final'] == w) & (~active['withdrew'])]
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())

        if len(elim) == 0 or is_finals:
            continue

        names = active['celebrity_name'].tolist()
        n = len(names)
        J = active[col].values.astype(float)

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
        bottom2 = set(np.argsort(-c)[:2])

        elim_names = elim['celebrity_name'].tolist()
        elim_idxs = [names.index(e) for e in elim_names if e in names]
        k = len(elim_idxs)

        if k == 1:
            total += 1
            if elim_idxs[0] in bottom2:
                correct += 1
        elif k == 2:
            total += 1
            if all(i in bottom2 for i in elim_idxs):
                correct += 1
        else:
            # skip if more than 2 eliminations in bottom2 regime
            continue

    return correct / total if total > 0 else 1.0


# =============================================================================
# Main
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

    print(f"{'Season':>6} | {'Regime':>7} | {'CSR':>6} | {'Certainty':>9}")
    print("-" * 40)

    for season in sorted(df['season'].unique()):
        sdf = df[df['season'] == season].copy()
        weeks = get_season_weeks(sdf)
        if not weeks:
            continue

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

        results, week_info = solve_func(sdf, weeks)
        csr = validate_func(results, sdf, weeks)
        stats = ensemble_solve(solve_func, sdf, weeks)

        avg_cert = np.mean([s['certainty'] for s in stats.values()])

        for w in weeks:
            col = f'J{w}'
            active = sdf[sdf[col] > 0]
            elim_names = active[(active['elim_week_final'] == w) & (~active['withdrew'])]['celebrity_name'].tolist()

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

    pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    mdf = pd.DataFrame(metrics)
    print("\nSummary by Regime:")
    for regime in ['rank', 'percent', 'bottom2']:
        rm = mdf[mdf['regime'] == regime]
        if len(rm) > 0:
            print(f"  {regime.upper()}: CSR={rm['csr'].mean():.1%}, Certainty={rm['avg_certainty'].mean():.3f}")


if __name__ == '__main__':
    main()
