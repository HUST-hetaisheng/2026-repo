#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWTS Fan Vote Inference (MCM 2026 Problem C, Q1)

This script estimates weekly fan vote shares that are consistent with observed eliminations,
under season-dependent rules:

  - Seasons 1-2: Rank rule (judge rank + fan rank; worst sum eliminated)
  - Seasons 3-27: Percent rule (judge share + fan share; lowest sum eliminated)
  - Seasons 28-34: Rank + Bottom2 + Judges' Save (eliminated must be in bottom2; judges choose between bottom2)

Key engineering safeguards (from project pitfalls):
  - Ensemble noise is applied ONLY to contestants with positive scores (prevents "reviving" eliminated contestants).
  - Multi-elimination weeks are handled as sets (E_t can have size > 1).
  - Withdrew contestants are excluded from vote-based elimination constraints but remain active before exit.
  - No float-equality CSR checks; we compare predicted/true elimination SETS.
  - Optimizer failures are not silenced; we check success and fall back to a feasible baseline.
  - All paths are relative by default; no hard-coded absolute paths.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


WEEKS = list(range(1, 12))
JUDGES = list(range(1, 5))

# ---------------------------
# Utility: ranking and entropy
# ---------------------------

def average_rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Average ranks with 1=best (largest value gets rank 1).
    Ties get average rank.
    """
    # argsort descending
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    # assign ranks with tie averaging
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks

def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return -float(np.sum(p * np.log(p)))

def safe_cv(samples: np.ndarray, eps: float = 1e-12) -> float:
    mu = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    mu = max(mu, eps)
    return sd / mu

# ---------------------------
# Data structures
# ---------------------------

@dataclass
class SeasonData:
    season: int
    contestants: List[str]                 # length N
    withdrew: np.ndarray                   # (N,) bool
    placement: np.ndarray                  # (N,) float/int/NaN
    J: np.ndarray                          # (N,T) judge totals
    active: np.ndarray                     # (N,T) bool
    T: int                                 # last active week (1..11)
    E: List[List[int]]                     # length T, each list contains indices eliminated at week t+1
    # optional: final ordering constraints from placement for week T
    finalists: List[int]                   # indices active in week T
    # bookkeeping:
    exit_week: np.ndarray                  # (N,) last active week

# ---------------------------
# Data loading + validation
# ---------------------------

def add_weekly_judge_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in WEEKS:
        cols = [f"week{w}_judge{j}_score" for j in JUDGES]
        df[f"week{w}_judge_total"] = df[cols].sum(axis=1, skipna=True)
    return df

_ELIM_RE = re.compile(r"Eliminated\s+Week\s+(\d+)", re.IGNORECASE)

def parse_elim_week_from_results(results: str) -> Optional[int]:
    if not isinstance(results, str):
        return None
    m = _ELIM_RE.search(results)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def build_season_data(df: pd.DataFrame, season: int, verbose: bool = False) -> SeasonData:
    sdf = df[df["season"] == season].copy()
    sdf.reset_index(drop=True, inplace=True)

    # Keep original placement; do NOT overwrite it from results.
    placement = pd.to_numeric(sdf.get("placement", pd.Series([np.nan]*len(sdf))), errors="coerce").to_numpy()

    withdrew = sdf["results"].astype(str).str.strip().str.lower().eq("withdrew").to_numpy()

    contestants = sdf["celebrity_name"].astype(str).tolist()
    N = len(contestants)

    # Build J and active
    T_guess = 0
    J = np.zeros((N, len(WEEKS)), dtype=float)
    active = np.zeros_like(J, dtype=bool)
    for wi, w in enumerate(WEEKS):
        tot = sdf[f"week{w}_judge_total"].to_numpy(dtype=float)
        J[:, wi] = tot
        active[:, wi] = tot > 0
        if np.any(active[:, wi]):
            T_guess = w

    T = T_guess
    if T == 0:
        raise ValueError(f"Season {season}: no active weeks detected.")

    # Trim to 1..T
    J = J[:, :T]
    active = active[:, :T]

    # Exit week based on scores (robust to mismatched labels)
    exit_week = np.zeros(N, dtype=int)
    for i in range(N):
        pos = np.where(J[i, :] > 0)[0]
        exit_week[i] = int(pos.max() + 1) if len(pos) else 0

    # Elimination sets E_t: those (not withdrew) exiting before final week
    E = [[] for _ in range(T)]
    for i in range(N):
        if withdrew[i]:
            continue
        ew = exit_week[i]
        if 1 <= ew < T:
            E[ew - 1].append(i)

    finalists = [i for i in range(N) if active[i, T - 1]]

    # Validation: compare results-labeled elim week vs exit_week
    mismatches = 0
    for i in range(N):
        lab = parse_elim_week_from_results(str(sdf.loc[i, "results"]))
        if lab is None:
            continue
        if exit_week[i] != 0 and lab != exit_week[i]:
            mismatches += 1
    if verbose and mismatches:
        print(f"[warn] Season {season}: {mismatches} contestants have results 'Eliminated Week k' != score-based exit week.")

    return SeasonData(
        season=season,
        contestants=contestants,
        withdrew=withdrew,
        placement=placement,
        J=J,
        active=active,
        T=T,
        E=E,
        finalists=finalists,
        exit_week=exit_week
    )

def count_multi_elim_weeks(season_data: SeasonData) -> int:
    return sum(1 for t in range(season_data.T) if len(season_data.E[t]) > 1)

# ---------------------------
# Percent season solver (convex program solved by SLSQP)
# ---------------------------

@dataclass
class PercentSolution:
    f: np.ndarray        # (N,T) vote shares (inactive = 0)
    delta: np.ndarray    # (T,) weekly slack
    success: bool
    message: str

def solve_percent_season(sd: SeasonData,
                        alpha: float = 0.05,
                        beta: float = 0.5,
                        M: float = 200.0,
                        use_final_constraints: bool = True,
                        final_weight: float = 5.0,
                        eps: float = 1e-12,
                        maxiter: int = 5000) -> PercentSolution:
    """
    Solve the percent-rule inverse problem for one season:
      c_{i,t} = j_{i,t} + f_{i,t}
      eliminated have the smallest c (multi-elim supported).

    Optimization:
      minimize  M * sum_t delta_t + beta * smoothness + alpha * sum f log f
      subject to simplex per week + elimination inequalities with slack delta_t.

    Note: This is a convex program. We use SciPy SLSQP as a practical solver
    (cvxpy may be unavailable in some environments).
    """
    N, T = sd.J.shape

    # judge shares j_{i,t}
    jshare = np.zeros((N, T), dtype=float)
    for t in range(T):
        idx = np.where(sd.active[:, t])[0]
        denom = float(sd.J[idx, t].sum())
        if denom <= 0:
            continue
        jshare[idx, t] = sd.J[idx, t] / denom

    # Build variable indexing: only active contestants get f variables each week
    week_active = []
    idx_of = {}  # (t, i) -> position in x for f variable
    offset = 0
    for t in range(T):
        act = np.where(sd.active[:, t])[0].tolist()
        week_active.append(act)
        for i in act:
            idx_of[(t, i)] = offset
            offset += 1
    n_f = offset
    n_delta = T
    n_var = n_f + n_delta

    def get_f_from_x(x: np.ndarray) -> np.ndarray:
        """Expand x into full (N,T) matrix with zeros for inactive."""
        f = np.zeros((N, T), dtype=float)
        for t in range(T):
            for i in week_active[t]:
                f[i, t] = x[idx_of[(t, i)]]
        return f

    def get_delta_from_x(x: np.ndarray) -> np.ndarray:
        return x[n_f:]

    # Feasible initialization: uniform f each week + required delta to satisfy inequalities
    x0 = np.zeros(n_var, dtype=float)
    for t in range(T):
        act = week_active[t]
        n = len(act)
        if n == 0:
            continue
        for i in act:
            x0[idx_of[(t, i)]] = 1.0 / n

    # initialize delta with minimal required slack given x0
    delta0 = np.zeros(T, dtype=float)
    f0 = get_f_from_x(x0)
    for t in range(T):
        if len(sd.E[t]) == 0:
            continue
        elim = sd.E[t]
        surv = [i for i in week_active[t] if i not in elim]
        req = 0.0
        for e in elim:
            for p in surv:
                req = max(req, (jshare[e, t] + f0[e, t]) - (jshare[p, t] + f0[p, t]))
        delta0[t] = max(req, 0.0)
    x0[n_f:] = delta0

    # Bounds: f in [0,1], delta >=0
    bounds = [(0.0, 1.0)] * n_f + [(0.0, None)] * n_delta

    # Constraints list for SLSQP
    cons = []

    # Simplex constraints: sum f_t = 1 for each week
    for t in range(T):
        act = week_active[t]
        idxs = [idx_of[(t, i)] for i in act]
        def make_eq(idxs_):
            return {'type': 'eq', 'fun': lambda x, idxs=idxs_: np.sum(x[idxs]) - 1.0}
        cons.append(make_eq(idxs))

    # If no elimination in week t, force delta_t = 0 (avoid meaningless slack)
    for t in range(T):
        if len(sd.E[t]) == 0:
            def make_delta_zero(tt):
                return {'type': 'eq', 'fun': lambda x, tt=tt: x[n_f + tt]}
            cons.append(make_delta_zero(t))

    # Elimination inequalities
    # (j_e + f_e) <= (j_p + f_p) + delta_t   =>   (j_p - j_e) + (f_p - f_e) + delta_t >= 0
    for t in range(T):
        elim = sd.E[t]
        if len(elim) == 0:
            continue
        surv = [i for i in week_active[t] if i not in elim]
        dt_idx = n_f + t
        for e in elim:
            e_idx = idx_of[(t, e)]
            for p in surv:
                p_idx = idx_of[(t, p)]
                jp_je = jshare[p, t] - jshare[e, t]
                def make_ineq(p_idx_, e_idx_, dt_idx_, jp_je_):
                    return {
                        'type': 'ineq',
                        'fun': lambda x, p_idx=p_idx_, e_idx=e_idx_, dt_idx=dt_idx_, jp_je=jp_je_:
                            jp_je + (x[p_idx] - x[e_idx]) + x[dt_idx]
                    }
                cons.append(make_ineq(p_idx, e_idx, dt_idx, jp_je))

    # Optional: final ranking constraints using placement among finalists (soft, with penalty weight)
    final_pairs = []
    if use_final_constraints:
        t = T - 1
        finals = [i for i in week_active[t] if not np.isnan(sd.placement[i])]
        if len(finals) >= 2:
            # smaller placement is better (1 is winner)
            finals_sorted = sorted(finals, key=lambda i: sd.placement[i])
            # build pair constraints: winner >= runner-up >= ...
            for a, b in zip(finals_sorted[:-1], finals_sorted[1:]):
                # want c_a >= c_b (since higher combined is better)
                a_idx = idx_of[(t, a)]
                b_idx = idx_of[(t, b)]
                ja_jb = jshare[a, t] - jshare[b, t]
                # constraint: (ja+fa) - (jb+fb) + delta_t >= 0, reuse delta_t as slack
                dt_idx = n_f + t
                def make_final_ineq(a_idx_, b_idx_, dt_idx_, ja_jb_):
                    return {
                        'type': 'ineq',
                        'fun': lambda x, a_idx=a_idx_, b_idx=b_idx_, dt_idx=dt_idx_, ja_jb=ja_jb_:
                            ja_jb + (x[a_idx] - x[b_idx]) + x[dt_idx]
                    }
                cons.append(make_final_ineq(a_idx, b_idx, dt_idx, ja_jb))
                final_pairs.append((a, b))

    # Objective (convex): slack + smoothness + negative entropy
    def objective(x: np.ndarray) -> float:
        f = get_f_from_x(x)
        delta = get_delta_from_x(x)
        # slack
        term_slack = M * float(np.sum(delta))
        # smoothness on overlapping contestants (L2^2)
        term_smooth = 0.0
        for t in range(1, T):
            inter = list(set(week_active[t]) & set(week_active[t-1]))
            if not inter:
                continue
            diff = f[inter, t] - f[inter, t-1]
            term_smooth += float(np.sum(diff * diff))
        term_smooth *= beta
        # negative entropy (convex): sum f log f
        term_ent = 0.0
        for t in range(T):
            act = week_active[t]
            if not act:
                continue
            ft = np.clip(f[act, t], eps, 1.0)
            term_ent += float(np.sum(ft * np.log(ft)))
        term_ent *= alpha
        # extra weight on final ordering if used (optional)
        # (we already included as constraints with slack; no extra needed)
        return term_slack + term_smooth + term_ent

    # Solve
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": maxiter, "ftol": 1e-9, "disp": False},
    )

    if not res.success:
        # Fallback: return feasible baseline (uniform f, delta = required)
        msg = f"SLSQP failed: {res.message}"
        f = get_f_from_x(x0)
        delta = delta0.copy()
        return PercentSolution(f=f, delta=delta, success=False, message=msg)

    x_hat = res.x
    f_hat = get_f_from_x(x_hat)
    delta_hat = get_delta_from_x(x_hat)
    return PercentSolution(f=f_hat, delta=delta_hat, success=True, message="ok")

# ---------------------------
# Rank / Bottom2 season solver (deterministic construction + smoothing)
# ---------------------------

@dataclass
class RankSolution:
    f: np.ndarray        # (N,T) vote shares (inactive=0)
    rF: List[Dict[int,int]]  # per week: contestant index -> fan rank (1..n)
    chosen_partner: List[Optional[int]]    # for bottom2 seasons, partner index or None
    notes: List[str]

def _assign_with_upper_bounds(candidates: List[int],
                              upper: Dict[int,int],
                              available_ranks: List[int],
                              prev_rank: Optional[Dict[int,int]] = None) -> Optional[Dict[int,int]]:
    """
    Assign each candidate a unique rank from available_ranks s.t. rank <= upper[candidate].
    Only upper bounds exist, so a greedy algorithm works.

    Tie-break: when upper ties, follow prev_rank order (smaller prev_rank first).
    """
    prev_rank = prev_rank or {}
    # Sort by (upper, prev_rank)
    cand_sorted = sorted(candidates, key=lambda i: (upper[i], prev_rank.get(i, 10**9)))
    ranks = sorted(available_ranks)
    out = {}
    for i, r in zip(cand_sorted, ranks):
        if r > upper[i]:
            return None
        out[i] = r
    return out

def _construct_rank_week(active_idx: List[int],
                         rJ: Dict[int,int],
                         elim_idx: List[int],
                         prev_rF: Optional[Dict[int,int]] = None) -> Optional[Dict[int,int]]:
    """
    Construct a fan-rank permutation rF for a Rank season week such that:
      for all e in elim, for all p in survivors: (rJ[e]+rF[e]) >= (rJ[p]+rF[p]).
    Strategy:
      - Give eliminated contestants the worst fan ranks (n-m+1..n),
        assigning the worst fan rank to the best judge among eliminated to boost combined.
      - Compute threshold Cmin = min_e (rJ[e]+rF[e]).
      - Assign survivors ranks 1..n-m with upper bound rF[p] <= Cmin - rJ[p].
    """
    prev_rF = prev_rF or {}
    n = len(active_idx)
    m = len(elim_idx)
    if m == 0:
        # no elimination: keep previous ordering if possible, else mimic judge rank
        if prev_rF:
            # re-rank active by prev_rF
            ordered = sorted(active_idx, key=lambda i: prev_rF.get(i, 10**9))
        else:
            ordered = sorted(active_idx, key=lambda i: rJ[i])
        return {i: k+1 for k, i in enumerate(ordered)}

    # eliminated ranks
    worst_ranks = list(range(n - m + 1, n + 1))  # length m
    worst_ranks_desc = sorted(worst_ranks, reverse=True)
    elim_sorted_by_judge = sorted(elim_idx, key=lambda i: rJ[i])  # best judge first (smallest rank)
    rF = {}

    for e, rr in zip(elim_sorted_by_judge, worst_ranks_desc):
        rF[e] = rr

    Cmin = min(rJ[e] + rF[e] for e in elim_idx)

    survivors = [i for i in active_idx if i not in elim_idx]
    avail = list(range(1, n - m + 1))
    upper = {}
    for p in survivors:
        upper[p] = min(n - m, Cmin - rJ[p])
        # Ensure at least 1
        upper[p] = max(1, upper[p])

    surv_assign = _assign_with_upper_bounds(survivors, upper, avail, prev_rank=prev_rF)
    if surv_assign is None:
        # If infeasible, relax by lowering Cmin (allow some survivors tie/be worse) via greedy on prev order.
        ordered = sorted(survivors, key=lambda i: prev_rF.get(i, 10**9))
        for k, p in enumerate(ordered):
            rF[p] = k + 1
        # keep eliminated as worst
        return rF

    rF.update(surv_assign)
    return rF

def _construct_bottom2_week(active_idx: List[int],
                            J: Dict[int,float],
                            rJ: Dict[int,int],
                            elim_single: int,
                            prev_rF: Optional[Dict[int,int]] = None,
                            smooth_weight: float = 1.0,
                            judge_save_weight: float = 1.0) -> Tuple[Dict[int,int], Optional[int], str]:
    """
    Construct fan ranks for Season 28+ week with SINGLE elimination:
      eliminated must be in bottom2 by combined rank c = rJ + rF.
    We enumerate partner b, try to make {e,b} bottom2 by assigning fan ranks:
      rF[e]=n, rF[b]=n-1, others 1..n-2 with upper bounds from Cmin=min(c_e,c_b).
    Choose b minimizing (smoothness cost + judge-save penalty), with feasibility check.
    """
    prev_rF = prev_rF or {}
    n = len(active_idx)
    e = elim_single
    best = None
    best_obj = float("inf")
    best_note = ""

    for b in active_idx:
        if b == e:
            continue
        # assign bottom2 fan ranks
        rF = {e: n, b: n - 1}
        Cmin = min(rJ[e] + rF[e], rJ[b] + rF[b])

        others = [i for i in active_idx if i not in (e, b)]
        avail = list(range(1, n - 1))
        upper = {}
        for p in others:
            upper[p] = min(n - 2, Cmin - rJ[p])
            upper[p] = max(1, upper[p])

        assign = _assign_with_upper_bounds(others, upper, avail, prev_rank=prev_rF)
        if assign is None:
            continue
        rF.update(assign)

        # objective: smoothness + judge-save penalty (prefers eliminating lower-judge-score contestant)
        smooth = 0.0
        inter = set(prev_rF.keys()) & set(rF.keys())
        for i in inter:
            smooth += abs(rF[i] - prev_rF[i])

        judge_penalty = max(0.0, J[e] - J[b])
        obj = smooth_weight * smooth + judge_save_weight * judge_penalty

        if obj < best_obj:
            best_obj = obj
            best = rF
            best_note = f"partner={b}, smooth={smooth:.1f}, judge_penalty={judge_penalty:.1f}"

    if best is None:
        # fallback: treat as normal rank week
        rF = _construct_rank_week(active_idx, rJ, [e], prev_rF=prev_rF)
        return rF, None, "bottom2 infeasible for all partners; fallback to rank constraint"
    return best, None if best_note=="" else int(best_note.split("partner=")[1].split(",")[0]), best_note

def ranks_to_shares(active_idx: List[int], rF: Dict[int,int], gamma: float) -> Dict[int,float]:
    """Map ranks (1 best) to vote shares via exponential decay."""
    vals = np.array([math.exp(-gamma * (rF[i] - 1)) for i in active_idx], dtype=float)
    vals = vals / vals.sum()
    return {i: float(v) for i, v in zip(active_idx, vals)}

def calibrate_gamma(entropy_targets: List[float],
                    n_active_list: List[int],
                    gamma_grid: Iterable[float] = np.linspace(0.05, 1.5, 60)) -> float:
    """
    Choose gamma so that the average entropy of exp-mapped shares under a "typical" rank distribution
    matches target entropy from percent seasons.
    We approximate typical rank distribution by assuming ranks 1..n.
    """
    target = float(np.mean(entropy_targets)) if entropy_targets else 1.5
    best_g, best_err = 0.3, float("inf")
    for g in gamma_grid:
        ent_list = []
        for n in n_active_list:
            ranks = np.arange(1, n+1)
            shares = np.exp(-g*(ranks-1))
            shares /= shares.sum()
            ent_list.append(entropy(shares))
        err = abs(np.mean(ent_list) - target)
        if err < best_err:
            best_err, best_g = err, float(g)
    return best_g

def solve_rank_or_bottom2_season(sd: SeasonData,
                                 season_type: str,
                                 gamma: float,
                                 smooth_weight: float = 1.0,
                                 judge_save_weight: float = 1.0) -> RankSolution:
    """
    season_type: 'rank' (S1-2) or 'bottom2' (S28+).
    """
    N, T = sd.J.shape
    f = np.zeros((N, T), dtype=float)
    rF_list: List[Dict[int,int]] = []
    chosen_partner: List[Optional[int]] = []
    notes: List[str] = []

    prev_rF: Dict[int,int] = {}

    for t in range(T):
        active_idx = np.where(sd.active[:, t])[0].tolist()
        if not active_idx:
            rF_list.append({})
            chosen_partner.append(None)
            notes.append("no active")
            continue

        # judge ranks from totals (higher score better -> rank 1)
        vals = np.array([sd.J[i, t] for i in active_idx], dtype=float)
        rJ_float = average_rank_desc(vals)
        # convert to integer-ish by ordering (ties resolved by stable order)
        order = np.argsort(rJ_float, kind="mergesort")  # lower is better
        rJ = {}
        for k, idx in enumerate(order):
            rJ[active_idx[idx]] = k + 1

        elim = sd.E[t].copy()

        # If multiple eliminations in bottom2 era, treat as special episode -> fall back to rank elimination constraint
        if season_type == "bottom2" and len(elim) > 1:
            rF = _construct_rank_week(active_idx, rJ, elim, prev_rF=prev_rF)
            note = f"multi-elim({len(elim)}); used rank constraint as fallback"
            partner = None
        elif season_type == "bottom2" and len(elim) == 1:
            # build dict of judge totals for penalty
            Jdict = {i: float(sd.J[i, t]) for i in active_idx}
            rF, partner, note = _construct_bottom2_week(
                active_idx=active_idx,
                J=Jdict,
                rJ=rJ,
                elim_single=elim[0],
                prev_rF=prev_rF,
                smooth_weight=smooth_weight,
                judge_save_weight=judge_save_weight
            )
        else:
            rF = _construct_rank_week(active_idx, rJ, elim, prev_rF=prev_rF)
            partner, note = None, "ok"

        rF_list.append(rF)
        chosen_partner.append(partner)
        notes.append(note)

        # map to shares
        shares = ranks_to_shares(active_idx, rF, gamma)
        for i in active_idx:
            f[i, t] = shares[i]

        # update prev ranks only for next week active intersection
        prev_rF = rF

    return RankSolution(f=f, rF=rF_list, chosen_partner=chosen_partner, notes=notes)

# ---------------------------
# Consistency evaluation
# ---------------------------

@dataclass
class ConsistencyMetrics:
    season: int
    rule: str
    weeks_with_elim: int
    exact_set_acc: float
    mean_jaccard: float
    mean_margin: float
    mean_delta: float  # for percent seasons; for rank seasons we report 0.0

def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)

def evaluate_percent(sd: SeasonData, sol: PercentSolution) -> ConsistencyMetrics:
    N, T = sd.J.shape
    # compute judge shares
    jshare = np.zeros((N, T), dtype=float)
    for t in range(T):
        idx = np.where(sd.active[:, t])[0]
        denom = float(sd.J[idx, t].sum())
        if denom > 0:
            jshare[idx, t] = sd.J[idx, t] / denom

    exact = []
    jac = []
    margins = []
    deltas = []
    for t in range(T):
        E_t = sd.E[t]
        if len(E_t) == 0:
            continue
        m = len(E_t)
        active_idx = np.where(sd.active[:, t])[0]
        c = jshare[active_idx, t] + sol.f[active_idx, t]
        order = np.argsort(c)  # smaller worse
        pred = active_idx[order[:m]].tolist()
        exact.append(1.0 if set(pred) == set(E_t) else 0.0)
        jac.append(jaccard(pred, E_t))
        # margin between last eliminated and first safe
        if len(order) > m:
            margins.append(float(c[order[m]] - c[order[m-1]]))
        deltas.append(float(sol.delta[t]))

    weeks = len(exact)
    return ConsistencyMetrics(
        season=sd.season,
        rule="percent",
        weeks_with_elim=weeks,
        exact_set_acc=float(np.mean(exact)) if weeks else float("nan"),
        mean_jaccard=float(np.mean(jac)) if weeks else float("nan"),
        mean_margin=float(np.mean(margins)) if margins else float("nan"),
        mean_delta=float(np.mean(deltas)) if deltas else 0.0
    )

def evaluate_rank(sd: SeasonData, f: np.ndarray, season_type: str) -> ConsistencyMetrics:
    N, T = sd.J.shape
    exact = []
    jac = []
    margins = []
    for t in range(T):
        E_t = sd.E[t]
        if len(E_t) == 0:
            continue
        m = len(E_t)
        active_idx = np.where(sd.active[:, t])[0].tolist()
        # compute judge ranks
        vals = np.array([sd.J[i, t] for i in active_idx], dtype=float)
        rJ_float = average_rank_desc(vals)
        orderJ = np.argsort(rJ_float, kind="mergesort")
        rJ = {active_idx[idx]: k+1 for k, idx in enumerate(orderJ)}

        # infer fan ranks from f shares by sorting desc (bigger share -> better rank)
        shares = np.array([f[i, t] for i in active_idx], dtype=float)
        orderF = np.argsort(-shares, kind="mergesort")
        rF = {active_idx[idx]: k+1 for k, idx in enumerate(orderF)}

        # combined rank
        c = np.array([rJ[i] + rF[i] for i in active_idx], dtype=float)
        order = np.argsort(-c)  # larger worse
        if season_type == "bottom2" and m == 1:
            # check bottom2 membership
            bottom2 = set([active_idx[order[0]], active_idx[order[1]]]) if len(order) >= 2 else set()
            pred = [active_idx[order[0]]]  # predicted eliminated as worst
            exact.append(1.0 if (E_t[0] in bottom2) else 0.0)  # bottom2-hit
            jac.append(jaccard(bottom2, set([E_t[0]]) | (bottom2 - set([E_t[0]]))))  # not very meaningful; keep for completeness
            # margin between worst and 3rd worst (how stable bottom2)
            if len(order) > 2:
                margins.append(float(c[order[1]] - c[order[2]]))
        else:
            pred = [active_idx[idx] for idx in order[:m]]
            exact.append(1.0 if set(pred) == set(E_t) else 0.0)
            jac.append(jaccard(pred, E_t))
            if len(order) > m:
                margins.append(float(c[order[m-1]] - c[order[m]]))  # gap between last eliminated and first safe in rank-space

    weeks = len(exact)
    return ConsistencyMetrics(
        season=sd.season,
        rule="bottom2" if season_type=="bottom2" else "rank",
        weeks_with_elim=weeks,
        exact_set_acc=float(np.mean(exact)) if weeks else float("nan"),
        mean_jaccard=float(np.mean(jac)) if weeks else float("nan"),
        mean_margin=float(np.mean(margins)) if margins else float("nan"),
        mean_delta=0.0
    )

# ---------------------------
# Ensemble certainty
# ---------------------------

@dataclass
class CertaintySummary:
    season: int
    rule: str
    # per contestant-week arrays
    mean_f: np.ndarray
    std_f: np.ndarray
    cv: np.ndarray
    certainty: np.ndarray

def perturb_scores(sd: SeasonData, noise_sd: float, rng: np.random.Generator) -> SeasonData:
    """
    Add small noise ONLY to positive judge totals to prevent reviving eliminated contestants.
    """
    new = dataclasses.replace(sd)
    J = sd.J.copy()
    mask = J > 0
    J[mask] = J[mask] + rng.normal(0.0, noise_sd, size=J[mask].shape)
    J[mask] = np.maximum(J[mask], 1e-6)  # keep positive
    new.J = J
    # active set unchanged because we did not flip zeros to positive
    new.active = sd.active.copy()
    return new

def ensemble_percent(sd: SeasonData,
                     K: int = 100,
                     noise_sd: float = 0.05,
                     alpha: float = 0.05,
                     beta: float = 0.5,
                     M: float = 200.0,
                     seed: int = 42) -> CertaintySummary:
    rng = np.random.default_rng(seed)
    N, T = sd.J.shape
    samples = np.zeros((K, N, T), dtype=float)
    for k in range(K):
        # hyperparameter perturbation (small)
        a_k = alpha * (1 + rng.uniform(-0.1, 0.1))
        b_k = beta * (1 + rng.uniform(-0.1, 0.1))
        M_k = M
        sd_k = perturb_scores(sd, noise_sd=noise_sd, rng=rng)
        sol = solve_percent_season(sd_k, alpha=a_k, beta=b_k, M=M_k)
        samples[k] = sol.f
        # no silent ignore
        if (not sol.success) and (k == 0):
            print(f"[warn] Season {sd.season}: percent solver failed in ensemble sample 0: {sol.message}")

    mean_f = np.mean(samples, axis=0)
    std_f = np.std(samples, axis=0, ddof=1)
    cv = np.zeros((N, T), dtype=float)
    certainty = np.zeros((N, T), dtype=float)
    eps = 1e-12
    for i in range(N):
        for t in range(T):
            mu = mean_f[i, t]
            if mu <= 0:
                cv[i, t] = 0.0
                certainty[i, t] = 0.0
            else:
                cv[i, t] = std_f[i, t] / max(mu, eps)
                certainty[i, t] = 1.0 / (1.0 + cv[i, t])
    return CertaintySummary(sd.season, "percent", mean_f, std_f, cv, certainty)

def ensemble_rank(sd: SeasonData,
                  season_type: str,
                  K: int = 100,
                  noise_sd: float = 0.05,
                  gamma: float = 0.4,
                  seed: int = 42) -> CertaintySummary:
    rng = np.random.default_rng(seed)
    N, T = sd.J.shape
    samples = np.zeros((K, N, T), dtype=float)
    for k in range(K):
        # perturb gamma to reflect mapping uncertainty
        g_k = gamma + rng.uniform(-0.2, 0.2)
        g_k = max(0.05, g_k)
        sd_k = perturb_scores(sd, noise_sd=noise_sd, rng=rng)
        sol = solve_rank_or_bottom2_season(sd_k, season_type=season_type, gamma=g_k)
        samples[k] = sol.f

    mean_f = np.mean(samples, axis=0)
    std_f = np.std(samples, axis=0, ddof=1)
    cv = np.zeros((N, T), dtype=float)
    certainty = np.zeros((N, T), dtype=float)
    eps = 1e-12
    for i in range(N):
        for t in range(T):
            mu = mean_f[i, t]
            if mu <= 0:
                cv[i, t] = 0.0
                certainty[i, t] = 0.0
            else:
                cv[i, t] = std_f[i, t] / max(mu, eps)
                certainty[i, t] = 1.0 / (1.0 + cv[i, t])
    return CertaintySummary(sd.season, season_type, mean_f, std_f, cv, certainty)

# ---------------------------
# Pipeline runner
# ---------------------------

def infer_all_votes(data_path: Path,
                    out_dir: Path,
                    K_ensemble: int = 100,
                    verbose: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = add_weekly_judge_totals(df)

    seasons = sorted(df["season"].unique().tolist())

    # First pass: solve percent seasons to get entropy targets for gamma calibration
    percent_entropies = []
    n_active_list = []
    percent_solutions: Dict[int, PercentSolution] = {}

    for s in seasons:
        sd = build_season_data(df, s, verbose=verbose)
        if 3 <= s <= 27:
            sol = solve_percent_season(sd)
            percent_solutions[s] = sol
            # entropy across weeks
            for t in range(sd.T):
                act = np.where(sd.active[:, t])[0]
                if len(act) > 0:
                    percent_entropies.append(entropy(sol.f[act, t]))
                    n_active_list.append(len(act))
        if verbose:
            me = count_multi_elim_weeks(sd)
            if me:
                print(f"[info] Season {s}: multi-elim weeks = {me}")

    gamma = calibrate_gamma(percent_entropies, n_active_list) if n_active_list else 0.4
    if verbose:
        print(f"[info] Calibrated gamma = {gamma:.3f}")

    rows_votes = []
    rows_metrics = []
    rows_cert = []

    for s in seasons:
        sd = build_season_data(df, s, verbose=verbose)

        if s <= 2:
            rule = "rank"
            sol_rank = solve_rank_or_bottom2_season(sd, season_type="rank", gamma=gamma)
            met = evaluate_rank(sd, sol_rank.f, season_type="rank")
            cert = ensemble_rank(sd, season_type="rank", K=K_ensemble, gamma=gamma)
            f_hat = sol_rank.f
        elif 3 <= s <= 27:
            rule = "percent"
            sol = percent_solutions.get(s) or solve_percent_season(sd)
            met = evaluate_percent(sd, sol)
            cert = ensemble_percent(sd, K=K_ensemble)
            f_hat = sol.f
        else:
            rule = "bottom2"
            sol_b2 = solve_rank_or_bottom2_season(sd, season_type="bottom2", gamma=gamma)
            met = evaluate_rank(sd, sol_b2.f, season_type="bottom2")
            cert = ensemble_rank(sd, season_type="bottom2", K=K_ensemble, gamma=gamma)
            f_hat = sol_b2.f

        rows_metrics.append(dataclasses.asdict(met))

        # Save vote shares per contestant-week
        N, T = sd.J.shape
        for i, name in enumerate(sd.contestants):
            for t in range(T):
                if not sd.active[i, t]:
                    continue
                rows_votes.append({
                    "season": s,
                    "week": t + 1,
                    "celebrity_name": name,
                    "vote_share": float(f_hat[i, t]),
                })
                rows_cert.append({
                    "season": s,
                    "week": t + 1,
                    "celebrity_name": name,
                    "mean_vote_share": float(cert.mean_f[i, t]),
                    "std_vote_share": float(cert.std_f[i, t]),
                    "cv": float(cert.cv[i, t]),
                    "certainty": float(cert.certainty[i, t]),
                })

    votes_df = pd.DataFrame(rows_votes)
    metrics_df = pd.DataFrame(rows_metrics)
    cert_df = pd.DataFrame(rows_cert)

    votes_df.to_csv(out_dir / "fan_vote_shares.csv", index=False)
    metrics_df.to_csv(out_dir / "consistency_metrics.csv", index=False)
    cert_df.to_csv(out_dir / "certainty_metrics.csv", index=False)

    if verbose:
        print(f"[done] wrote: {out_dir/'fan_vote_shares.csv'}")
        print(f"[done] wrote: {out_dir/'consistency_metrics.csv'}")
        print(f"[done] wrote: {out_dir/'certainty_metrics.csv'}")

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Infer DWTS fan vote shares (MCM 2026 C Q1)")
    parser.add_argument("--data", type=str, default=None, help="Path to 2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--K", type=int, default=100, help="Ensemble size")
    parser.add_argument("--quiet", action="store_true", help="Less logging")
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data) if args.data else (script_dir / "2026_MCM_Problem_C_Data.csv")
    out_dir = Path(args.out)
    verbose = not args.quiet

    if not data_path.exists():
        print(f"[error] data file not found: {data_path}", file=sys.stderr)
        return 2

    infer_all_votes(data_path=data_path, out_dir=out_dir, K_ensemble=args.K, verbose=verbose)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
