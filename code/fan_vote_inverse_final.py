
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize, linprog
from scipy.stats import rankdata

DATA_PATH = r"D:\2026-repo\data\2026_MCM_Problem_C_Data.csv"
OUT_DIR = Path(r"D:\2026-repo\data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

weeks = list(range(1, 12))


def load_data(path):
    df = pd.read_csv(path)
    for w in weeks:
        cols = [c for c in df.columns if c.startswith(f"week{w}_judge")]
        if cols:
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df[f"J_week{w}"] = df[cols].sum(axis=1, skipna=True)
    df["elim_week"] = df["results"].str.extract(r"Eliminated Week (\d+)")[0].astype(float)
    df["withdrew"] = df["results"].str.contains("Withdrew", case=False, na=False)
    return df


def last_active_week(row):
    lw = 0
    for w in weeks:
        col = f"J_week{w}"
        if col in row and pd.notna(row[col]) and row[col] > 0:
            lw = w
    return lw


def season_regime(season):
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom2"


def build_weekly_data(season_df):
    season_df = season_df.copy()
    season_df["last_week"] = season_df.apply(last_active_week, axis=1)
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
        withdrew = active.loc[(active["elim_week"] == w) & (active["withdrew"]), "celebrity_name"].tolist()
        if w == max_week:
            week_type = "finals"
        elif len(eliminated) == 0:
            week_type = "withdrew_only" if len(withdrew) > 0 else "none"
        elif len(eliminated) == 1:
            week_type = "single"
        else:
            week_type = "multi"
        weekly[w] = {
            "names": names,
            "J": J,
            "eliminated": eliminated,
            "withdrew": withdrew,
            "week_type": week_type,
            "placement": active.set_index("celebrity_name")["placement"].to_dict()
        }
    final_df = season_df[season_df["placement"].notna()].copy()
    final_df = final_df.sort_values("placement")
    final_ranking = final_df["celebrity_name"].tolist()
    return weekly, final_ranking, max_week


def feasible_f_percent(j_share, elim_idx, non_elim_idx):
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


def solve_percent_week(J, names, eliminated, placement_map, f_prev=None,
                       lam1=1.0, lam2=0.5, rho=0.5):
    n = len(J)
    J = np.array(J, dtype=float)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n

    if f_prev is None or len(f_prev) != n:
        f_prev = j_share.copy()

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

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 800})
    if not res.success:
        f = x0
    else:
        f = np.clip(res.x, 0, None)
        f = f / f.sum() if f.sum() > 0 else np.ones(n) / n
    return f


def enforce_elimination_rank(names, J, eliminated, fan_ranks):
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
            # swap eliminated with the worse of bottom2
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


def assign_fan_ranks_rank(names, J, eliminated, placement_map, prev_rank_map=None):
    n = len(names)
    j_rank = rankdata(-J, method="dense")

    if prev_rank_map:
        pref_order = sorted(range(n), key=lambda i: prev_rank_map.get(names[i], j_rank[i]))
    else:
        pref_order = list(np.argsort(j_rank))

    fan_ranks = np.zeros(n, dtype=int)

    if len(eliminated) > 0:
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]

        def place_key(idx):
            nm = names[idx]
            p = placement_map.get(nm)
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

    fan_ranks = enforce_elimination_bottom2(names, J, eliminated, fan_ranks)
    return fan_ranks


def assign_fan_ranks_bottom2(names, J, eliminated, placement_map, prev_rank_map=None):
    n = len(names)
    j_rank = rankdata(-J, method="dense")

    if prev_rank_map:
        pref_order = sorted(range(n), key=lambda i: prev_rank_map.get(names[i], j_rank[i]))
    else:
        pref_order = list(np.argsort(j_rank))

    fan_ranks = np.zeros(n, dtype=int)
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    if eliminated:
        elim_idx = name_to_idx[eliminated[0]]
        candidates = [i for i in range(n) if i != elim_idx]
        candidates = sorted(candidates, key=lambda i: -J[i])
        partner_idx = None
        for i in candidates:
            if J[i] >= J[elim_idx]:
                partner_idx = i
                break
        if partner_idx is None and candidates:
            partner_idx = candidates[0]

        fan_ranks[elim_idx] = n
        if partner_idx is not None:
            fan_ranks[partner_idx] = n - 1

    used = set(fan_ranks[fan_ranks > 0])
    for idx in pref_order:
        if fan_ranks[idx] > 0:
            continue
        for r in range(1, n + 1):
            if r not in used:
                fan_ranks[idx] = r
                used.add(r)
                break

    fan_ranks = enforce_elimination_rank(names, J, eliminated, fan_ranks)
    return fan_ranks


def consistency_week_percent(names, J, f, eliminated, placement_map):
    if not eliminated:
        return None
    j_share = J / J.sum() if J.sum() > 0 else np.ones(len(J)) / len(J)
    S = j_share + f
    k = len(eliminated)
    thresh = np.sort(S)[k - 1]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    # set consistency
    set_ok = all(S[name_to_idx[e]] <= thresh + 1e-9 for e in eliminated if e in name_to_idx)
    if not set_ok:
        return 0
    # internal order consistency for multi-elimination (only within E_w)
    if k > 1:
        for a in eliminated:
            for b in eliminated:
                if a == b:
                    continue
                pa = placement_map.get(a)
                pb = placement_map.get(b)
                if pd.notna(pa) and pd.notna(pb) and pa > pb:
                    ia = name_to_idx[a]
                    ib = name_to_idx[b]
                    if S[ia] > S[ib] + 1e-9:
                        return 0
    return 1


def consistency_week_rank(names, J, f, eliminated, placement_map):
    if not eliminated:
        return None
    j_rank = rankdata(-J, method="dense")
    f_rank = rankdata(-f, method="dense")
    R = j_rank + f_rank
    k = len(eliminated)
    thresh = np.sort(R)[-k]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    set_ok = all(R[name_to_idx[e]] >= thresh - 1e-9 for e in eliminated if e in name_to_idx)
    if not set_ok:
        return 0
    if k > 1:
        for a in eliminated:
            for b in eliminated:
                if a == b:
                    continue
                pa = placement_map.get(a)
                pb = placement_map.get(b)
                if pd.notna(pa) and pd.notna(pb) and pa > pb:
                    ia = name_to_idx[a]
                    ib = name_to_idx[b]
                    if R[ia] < R[ib] - 1e-9:
                        return 0
    return 1


def consistency_week_bottom2(names, J, f, eliminated, placement_map):
    if not eliminated:
        return None
    j_rank = rankdata(-J, method="dense")
    f_rank = rankdata(-f, method="dense")
    R = j_rank + f_rank
    k = len(eliminated)
    if k == 1:
        thresh = np.sort(R)[-2]
    else:
        thresh = np.sort(R)[-k]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    set_ok = all(R[name_to_idx[e]] >= thresh - 1e-9 for e in eliminated if e in name_to_idx)
    if not set_ok:
        return 0
    if k > 1:
        for a in eliminated:
            for b in eliminated:
                if a == b:
                    continue
                pa = placement_map.get(a)
                pb = placement_map.get(b)
                if pd.notna(pa) and pd.notna(pb) and pa > pb:
                    ia = name_to_idx[a]
                    ib = name_to_idx[b]
                    if R[ia] < R[ib] - 1e-9:
                        return 0
    return 1


def consistency_finals_percent(names, J, f, placement_map):
    j_share = J / J.sum() if J.sum() > 0 else np.ones(len(J)) / len(J)
    S = j_share + f
    # all pairwise orderings must match placement
    for i, ni in enumerate(names):
        pi = placement_map.get(ni)
        if not pd.notna(pi):
            continue
        for j, nj in enumerate(names):
            if i == j:
                continue
            pj = placement_map.get(nj)
            if not pd.notna(pj) or pi == pj:
                continue
            if pi < pj and S[i] + 1e-9 < S[j]:
                return 0
    return 1


def consistency_finals_rank(names, J, f, placement_map):
    j_rank = rankdata(-J, method="dense")
    f_rank = rankdata(-f, method="dense")
    R = j_rank + f_rank
    for i, ni in enumerate(names):
        pi = placement_map.get(ni)
        if not pd.notna(pi):
            continue
        for j, nj in enumerate(names):
            if i == j:
                continue
            pj = placement_map.get(nj)
            if not pd.notna(pj) or pi == pj:
                continue
            if pi < pj and R[i] - 1e-9 > R[j]:
                return 0
    return 1


def run_percent_ensemble(weekly, lam1, lam2, rho, n_ens=10):
    samples = defaultdict(list)
    for _ in range(n_ens):
        lam1_k = lam1 * (1 + np.random.uniform(-0.3, 0.3))
        lam2_k = lam2 * (1 + np.random.uniform(-0.3, 0.3))
        rho_k = rho * (1 + np.random.uniform(-0.3, 0.3))
        prev_map = None
        for w in sorted(weekly.keys()):
            data = weekly[w]
            names = data["names"]
            J = np.maximum(1e-3, data["J"] + np.random.normal(0, 0.5, len(data["J"])))
            eliminated = data["eliminated"] if data["week_type"] not in ["none", "withdrew_only"] else []
            placement_map = data["placement"]

            if prev_map:
                f_prev = np.array([prev_map.get(nm, np.nan) for nm in names], dtype=float)
                if np.isnan(f_prev).any():
                    j_share = J / J.sum() if J.sum() > 0 else np.ones(len(J)) / len(J)
                    f_prev = np.where(np.isnan(f_prev), j_share, f_prev)
                f_prev = f_prev / f_prev.sum() if f_prev.sum() > 0 else None
            else:
                f_prev = None

            f = solve_percent_week(J, names, eliminated, placement_map,
                                   f_prev=f_prev, lam1=lam1_k, lam2=lam2_k, rho=rho_k)
            prev_map = {nm: f[i] for i, nm in enumerate(names)}
            for i, nm in enumerate(names):
                samples[(nm, w)].append(f[i])
    return samples


def run_rank_ensemble(weekly, gamma, regime, n_ens=10):
    samples = defaultdict(list)
    for _ in range(n_ens):
        gamma_k = gamma * (1 + np.random.uniform(-0.4, 0.4))
        prev_rank_map = None
        for w in sorted(weekly.keys()):
            data = weekly[w]
            names = data["names"]
            J = np.maximum(1e-3, data["J"] + np.random.normal(0, 1.0, len(data["J"])))
            eliminated = data["eliminated"]
            placement_map = data["placement"]
            if regime == "rank":
                f_ranks = assign_fan_ranks_rank(names, J, eliminated, placement_map, prev_rank_map)
            else:
                f_ranks = assign_fan_ranks_bottom2(names, J, eliminated, placement_map, prev_rank_map)
            f = np.exp(-gamma_k * (f_ranks - 1))
            f = f / f.sum()
            prev_rank_map = {nm: f_ranks[i] for i, nm in enumerate(names)}
            for i, nm in enumerate(names):
                samples[(nm, w)].append(f[i])
    return samples


def summarize_samples(samples):
    out = {}
    for key, vals in samples.items():
        v = np.array(vals)
        mean = v.mean()
        std = v.std()
        cv = std / mean if mean > 0 else np.nan
        ci_low = np.percentile(v, 2.5)
        ci_high = np.percentile(v, 97.5)
        out[key] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width": ci_high - ci_low
        }
    return out


def main():
    df = load_data(DATA_PATH)
    results = []
    cons_rows = []

    lam1 = 1.0
    lam2 = 0.5
    rho = 0.5
    gamma = 0.3
    n_ens = 10

    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season]
        weekly, final_ranking, max_week = build_weekly_data(sdf)
        regime = season_regime(season)
        prev_f_map = None
        prev_rank_map = None

        # uncertainty samples per season
        if regime == "percent":
            samples = run_percent_ensemble(weekly, lam1, lam2, rho, n_ens=n_ens)
        else:
            samples = run_rank_ensemble(weekly, gamma, regime, n_ens=n_ens)
        stats = summarize_samples(samples)

        for w in sorted(weekly.keys()):
            data = weekly[w]
            names = data["names"]
            J = data["J"]
            eliminated = data["eliminated"]
            placement_map = data["placement"]
            week_type = data["week_type"]

            if regime == "percent":
                if prev_f_map:
                    f_prev = np.array([prev_f_map.get(nm, np.nan) for nm in names], dtype=float)
                    if np.isnan(f_prev).any():
                        j_share = J / J.sum() if J.sum() > 0 else np.ones(len(J)) / len(J)
                        f_prev = np.where(np.isnan(f_prev), j_share, f_prev)
                    f_prev = f_prev / f_prev.sum() if f_prev.sum() > 0 else None
                else:
                    f_prev = None

                if week_type in ["none", "withdrew_only"]:
                    f = solve_percent_week(J, names, [], placement_map, f_prev=f_prev,
                                           lam1=lam1, lam2=lam2, rho=0.0)
                else:
                    f = solve_percent_week(J, names, eliminated, placement_map, f_prev=f_prev,
                                           lam1=lam1, lam2=lam2, rho=rho)
                prev_f_map = {nm: f[i] for i, nm in enumerate(names)}
                cons = consistency_week_percent(names, J, f, eliminated, placement_map)

            elif regime == "rank":
                f_ranks = assign_fan_ranks_rank(names, J, eliminated, placement_map, prev_rank_map)
                f = np.exp(-gamma * (f_ranks - 1))
                f = f / f.sum()
                prev_rank_map = {nm: f_ranks[i] for i, nm in enumerate(names)}
                cons = consistency_week_rank(names, J, f, eliminated, placement_map)

            else:
                f_ranks = assign_fan_ranks_bottom2(names, J, eliminated, placement_map, prev_rank_map)
                f = np.exp(-gamma * (f_ranks - 1))
                f = f / f.sum()
                prev_rank_map = {nm: f_ranks[i] for i, nm in enumerate(names)}
                cons = consistency_week_bottom2(names, J, f, eliminated, placement_map)

            if week_type == "finals":
                if regime == "percent":
                    cons_val = str(consistency_finals_percent(names, J, f, placement_map))
                else:
                    cons_val = str(consistency_finals_rank(names, J, f, placement_map))
            elif week_type in ["none", "withdrew_only"] or not eliminated:
                # no vote-based elimination to compare; treat as consistent with "no-elim"
                cons_val = "1"
            else:
                cons_val = str(cons)
            cons_rows.append({"season": season, "week": w, "regime": regime, "consistent": cons_val})

            for i, nm in enumerate(names):
                st = stats.get((nm, w), {})
                results.append({
                    "season": season,
                    "week": w,
                    "celebrity_name": nm,
                    "fan_vote_share": f[i],
                    "fan_vote_mean": st.get("mean", f[i]),
                    "fan_vote_std": st.get("std", np.nan),
                    "cv": st.get("cv", np.nan),
                    "ci_low": st.get("ci_low", np.nan),
                    "ci_high": st.get("ci_high", np.nan),
                    "ci_width": st.get("ci_width", np.nan),
                    "judge_total": J[i],
                    "regime": regime,
                    "eliminated_this_week": nm in eliminated
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "fan_vote_results_final.csv", index=False)

    cons_df = pd.DataFrame(cons_rows)
    cons_df.to_csv(OUT_DIR / "consistency_by_week_final.csv", index=False)

    print("Saved:", OUT_DIR / "fan_vote_results_final.csv")
    print("Saved:", OUT_DIR / "consistency_by_week_final.csv")


if __name__ == "__main__":
    main()
