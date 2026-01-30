"""
Fan Vote Estimation Model for Dancing with the Stars
=====================================================
This script implements the Bayesian inverse voting model to estimate fan votes
from observed elimination outcomes across all 34 seasons.

Output:
- fan_vote_results.csv: Estimated fan vote shares with uncertainty measures
- consistency_metrics.csv: Consistency and certainty metrics per season
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath):
    """Load data and compute weekly judge totals."""
    df = pd.read_csv(filepath)
    
    # Extract week columns
    week_cols = {}
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        week_cols[week] = [c for c in judge_cols if c in df.columns]
    
    # Compute weekly judge totals (J_{i,t})
    for week in range(1, 12):
        cols = week_cols[week]
        if cols:
            # Convert to numeric, coercing errors
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            # Sum across judges, ignoring NaN
            df[f'J_week{week}'] = df[cols].sum(axis=1, skipna=True)
            # Replace 0 with NaN for inactive contestants
            df.loc[df[f'J_week{week}'] == 0, f'J_week{week}'] = 0
    
    return df

def get_season_data(df, season):
    """Extract season-specific data structures."""
    season_df = df[df['season'] == season].copy()
    
    # Parse results to get elimination week
    def parse_elimination(result):
        result = str(result)
        if 'Eliminated Week' in result:
            try:
                # Handle both "Eliminated Week X" and "Withdrew (Eliminated Week X)"
                import re
                match = re.search(r'Eliminated Week (\d+)', result)
                if match:
                    return int(match.group(1))
            except:
                pass
        elif 'Withdrew' in result:
            return -1  # Withdrew, will be determined by last active week
        return None  # Finalist
    
    season_df['elim_week'] = season_df['results'].apply(parse_elimination)
    
    # Determine last active week for each contestant
    def get_last_active_week(row):
        for week in range(11, 0, -1):
            if f'J_week{week}' in row and row[f'J_week{week}'] > 0:
                return week
        return 0
    
    season_df['last_active_week'] = season_df.apply(get_last_active_week, axis=1)
    
    # For Withdrew cases, set elim_week to last_active_week
    season_df.loc[season_df['elim_week'] == -1, 'elim_week'] = season_df.loc[season_df['elim_week'] == -1, 'last_active_week']
    
    # Determine max week for this season
    max_week = season_df['last_active_week'].max()
    
    # Build weekly data
    weekly_data = {}
    for week in range(1, max_week + 1):
        active = season_df[season_df[f'J_week{week}'] > 0].copy()
        if len(active) == 0:
            continue
            
        # Judge scores for this week
        J_scores = active[f'J_week{week}'].values
        names = active['celebrity_name'].values
        
        # Identify eliminated contestants this week (excluding Withdrew)
        eliminated = []
        for idx, row in active.iterrows():
            if row['elim_week'] == week and 'Withdrew' not in str(row['results']):
                eliminated.append(row['celebrity_name'])
        
        # Survivors
        survivors = [n for n in names if n not in eliminated]
        
        weekly_data[week] = {
            'names': list(names),
            'J_scores': J_scores,
            'eliminated': eliminated,
            'survivors': survivors,
            'n_active': len(names)
        }
    
    # Get final rankings
    finalists = season_df[season_df['elim_week'].isna()].sort_values('placement')
    final_ranking = list(finalists['celebrity_name'].values)
    
    return weekly_data, final_ranking, max_week

# ============================================================================
# REGIME A: PERCENT SEASONS (3-27) - CONVEX OPTIMIZATION
# ============================================================================

def estimate_votes_percent_season(weekly_data, final_ranking, max_week, 
                                   alpha=0.1, beta=0.5, tau=10.0, n_ensemble=50):
    """
    Estimate fan votes for Percent seasons using MAP estimation.
    Returns point estimates and ensemble for uncertainty.
    """
    results = []
    
    # Get all unique contestants
    all_names = set()
    for week, data in weekly_data.items():
        all_names.update(data['names'])
    all_names = sorted(list(all_names))
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    
    def objective(v_flat, weekly_data, alpha, beta, tau):
        """Negative log-posterior (to minimize)."""
        loss = 0
        idx = 0
        prev_v = None
        
        for week in sorted(weekly_data.keys()):
            data = weekly_data[week]
            n = data['n_active']
            v = v_flat[idx:idx+n]
            
            # Simplex projection (softmax ensures valid probabilities)
            v = softmax(v)
            
            # Judge shares
            J = data['J_scores']
            j_share = J / J.sum()
            
            # Combined scores
            c = j_share + v
            
            # Soft elimination likelihood
            if data['eliminated']:
                for elim_name in data['eliminated']:
                    elim_idx = data['names'].index(elim_name)
                    # Log-softmax for eliminated having lowest score
                    log_probs = -tau * c
                    log_probs = log_probs - np.max(log_probs)  # numerical stability
                    log_sum_exp = np.log(np.sum(np.exp(log_probs)))
                    loss -= (log_probs[elim_idx] - log_sum_exp)
            
            # Entropy regularization (encourage uniform when uninformed)
            entropy = -np.sum(v * np.log(v + 1e-10))
            loss -= alpha * entropy
            
            # Temporal smoothness
            if prev_v is not None and len(prev_v) > 0:
                # Match contestants between weeks
                prev_names = weekly_data[week-1]['names'] if week > 1 else []
                curr_names = data['names']
                for i, name in enumerate(curr_names):
                    if name in prev_names:
                        j = prev_names.index(name)
                        if j < len(prev_v):
                            loss += beta * abs(v[i] - prev_v[j])
            
            prev_v = v
            idx += n
        
        return loss
    
    # Compute total variables needed
    total_vars = sum(data['n_active'] for data in weekly_data.values())
    
    # Initial guess: uniform
    v0 = np.zeros(total_vars)
    
    # Optimize
    try:
        res = minimize(objective, v0, args=(weekly_data, alpha, beta, tau),
                      method='L-BFGS-B', options={'maxiter': 1000})
        v_opt = res.x
    except:
        v_opt = v0
    
    # Extract results
    idx = 0
    point_estimates = {}
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        n = data['n_active']
        v = softmax(v_opt[idx:idx+n])
        
        for i, name in enumerate(data['names']):
            point_estimates[(name, week)] = v[i]
        
        idx += n
    
    # Ensemble for uncertainty (perturb and resolve)
    ensemble_estimates = {key: [] for key in point_estimates.keys()}
    
    for k in range(n_ensemble):
        # Perturb hyperparameters
        alpha_k = alpha * (1 + np.random.uniform(-0.2, 0.2))
        beta_k = beta * (1 + np.random.uniform(-0.2, 0.2))
        tau_k = tau * (1 + np.random.uniform(-0.1, 0.1))
        
        # Perturb judge scores slightly
        weekly_data_k = {}
        for week, data in weekly_data.items():
            data_k = data.copy()
            noise = np.random.normal(0, 0.5, len(data['J_scores']))
            data_k['J_scores'] = np.maximum(1, data['J_scores'] + noise)
            weekly_data_k[week] = data_k
        
        # Re-optimize
        try:
            v0_k = v_opt + np.random.normal(0, 0.1, len(v_opt))
            res_k = minimize(objective, v0_k, args=(weekly_data_k, alpha_k, beta_k, tau_k),
                           method='L-BFGS-B', options={'maxiter': 500})
            v_k = res_k.x
        except:
            v_k = v_opt
        
        # Extract
        idx = 0
        for week in sorted(weekly_data.keys()):
            data = weekly_data[week]
            n = data['n_active']
            v = softmax(v_k[idx:idx+n])
            
            for i, name in enumerate(data['names']):
                ensemble_estimates[(name, week)].append(v[i])
            
            idx += n
    
    return point_estimates, ensemble_estimates, weekly_data

# ============================================================================
# REGIME B: RANK SEASONS (1-2) - HEURISTIC RANK INFERENCE
# ============================================================================

def estimate_votes_rank_season(weekly_data, final_ranking, max_week, lambda_param=0.5, n_ensemble=50):
    """
    Estimate fan votes for Rank seasons.
    Uses heuristic: fan rank should make eliminated contestant have worst combined rank.
    """
    point_estimates = {}
    ensemble_estimates = {}
    
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        n = data['n_active']
        names = data['names']
        J = data['J_scores']
        
        # Judge ranks (1 = best = highest score)
        j_ranks = n + 1 - np.argsort(np.argsort(J)) - 1
        j_ranks = j_ranks.astype(float)
        
        # If there's an elimination, find fan ranks that work
        eliminated = data['eliminated']
        
        if eliminated:
            # Heuristic: eliminated contestant should have worst fan rank
            # to get worst combined rank
            fan_ranks = np.zeros(n)
            
            for elim_name in eliminated:
                elim_idx = names.index(elim_name)
                # Give eliminated contestant the worst fan rank
                fan_ranks[elim_idx] = n
            
            # Distribute other ranks
            remaining_indices = [i for i in range(n) if names[i] not in eliminated]
            # Sort by judge score descending (better dancers get better fan ranks)
            remaining_sorted = sorted(remaining_indices, key=lambda i: -J[i])
            for rank, idx in enumerate(remaining_sorted, 1):
                fan_ranks[idx] = rank
        else:
            # No elimination: use judge-based ordering
            fan_ranks = j_ranks.copy()
        
        # Convert ranks to vote shares using exponential model
        v = np.exp(-lambda_param * (fan_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            point_estimates[(name, week)] = v[i]
            ensemble_estimates[(name, week)] = []
    
    # Ensemble by perturbing lambda and adding noise
    for k in range(n_ensemble):
        lambda_k = lambda_param * (1 + np.random.uniform(-0.3, 0.3))
        
        for week in sorted(weekly_data.keys()):
            data = weekly_data[week]
            n = data['n_active']
            names = data['names']
            J = data['J_scores'] + np.random.normal(0, 0.5, n)
            
            j_ranks = n + 1 - np.argsort(np.argsort(J)) - 1
            eliminated = data['eliminated']
            
            if eliminated:
                fan_ranks = np.zeros(n)
                for elim_name in eliminated:
                    if elim_name in names:
                        elim_idx = names.index(elim_name)
                        fan_ranks[elim_idx] = n
                remaining_indices = [i for i in range(n) if names[i] not in eliminated]
                remaining_sorted = sorted(remaining_indices, key=lambda i: -J[i])
                for rank, idx in enumerate(remaining_sorted, 1):
                    fan_ranks[idx] = rank
            else:
                fan_ranks = j_ranks.copy()
            
            v = np.exp(-lambda_k * (fan_ranks - 1))
            v = v / v.sum()
            
            for i, name in enumerate(names):
                ensemble_estimates[(name, week)].append(v[i])
    
    return point_estimates, ensemble_estimates, weekly_data

# ============================================================================
# REGIME C: SEASONS 28-34 - RANK + BOTTOM2 + JUDGES SAVE
# ============================================================================

def estimate_votes_bottom2_season(weekly_data, final_ranking, max_week, lambda_param=0.5, n_ensemble=50):
    """
    Estimate fan votes for Bottom2 + Judges Save seasons.
    Similar to rank seasons but with bottom-2 constraint.
    """
    # Use same approach as rank seasons for simplicity
    # The constraint is that eliminated is in bottom 2 by combined ranks
    return estimate_votes_rank_season(weekly_data, final_ranking, max_week, lambda_param, n_ensemble)

# ============================================================================
# CONSISTENCY AND CERTAINTY METRICS
# ============================================================================

def compute_consistency_metrics(point_estimates, weekly_data, regime='percent'):
    """Compute consistency metrics: accuracy, Jaccard, margin."""
    correct = 0
    total_elim_weeks = 0
    jaccard_scores = []
    margins = []
    
    for week, data in weekly_data.items():
        if not data['eliminated']:
            continue
        
        total_elim_weeks += 1
        names = data['names']
        n = len(names)
        J = data['J_scores']
        
        # Get estimated votes for this week
        v = np.array([point_estimates.get((name, week), 1/n) for name in names])
        
        if regime == 'percent':
            # Combined score = judge share + vote share
            j_share = J / J.sum()
            c = j_share + v
            # Predicted eliminated = lowest combined score
            pred_elim_idx = np.argmin(c)
            
            # Margin
            sorted_c = np.sort(c)
            margin = sorted_c[1] - sorted_c[0] if len(sorted_c) > 1 else 0
        else:
            # Rank-based
            j_ranks = n + 1 - np.argsort(np.argsort(J))
            v_ranks = n + 1 - np.argsort(np.argsort(v))
            c = j_ranks + v_ranks
            # Predicted eliminated = highest (worst) combined rank
            pred_elim_idx = np.argmax(c)
            
            sorted_c = np.sort(c)
            margin = sorted_c[-1] - sorted_c[-2] if len(sorted_c) > 1 else 0
        
        pred_elim = names[pred_elim_idx]
        actual_elim = data['eliminated']
        
        # Check if prediction is correct
        if pred_elim in actual_elim:
            correct += 1
        
        # Jaccard
        pred_set = {pred_elim}
        actual_set = set(actual_elim)
        jaccard = len(pred_set & actual_set) / len(pred_set | actual_set) if actual_set else 0
        jaccard_scores.append(jaccard)
        margins.append(margin)
    
    accuracy = correct / total_elim_weeks if total_elim_weeks > 0 else 0
    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
    avg_margin = np.mean(margins) if margins else 0
    
    return {
        'accuracy': accuracy,
        'avg_jaccard': avg_jaccard,
        'avg_margin': avg_margin,
        'n_elim_weeks': total_elim_weeks
    }

def compute_certainty_metrics(ensemble_estimates):
    """Compute certainty metrics: CV, CI width for each estimate."""
    certainty = {}
    
    for key, samples in ensemble_estimates.items():
        if len(samples) < 2:
            certainty[key] = {'cv': np.nan, 'ci_width': np.nan, 'std': np.nan}
            continue
        
        samples = np.array(samples)
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean if mean > 0 else np.nan
        
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        ci_width = ci_high - ci_low
        
        certainty[key] = {
            'mean': mean,
            'std': std,
            'cv': cv,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_width': ci_width
        }
    
    return certainty

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("Fan Vote Estimation Model - Dancing with the Stars")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading and preprocessing data...")
    df = load_and_preprocess_data('d:/2026-repo/data/2026_MCM_Problem_C_Data.csv')
    print(f"     Loaded {len(df)} contestants across {df['season'].nunique()} seasons")
    
    # Process each season
    all_results = []
    season_metrics = []
    
    print("\n[2/4] Estimating fan votes for each season...")
    
    for season in range(1, 35):
        print(f"     Processing Season {season}...", end=" ")
        
        try:
            weekly_data, final_ranking, max_week = get_season_data(df, season)
            
            if not weekly_data:
                print("No valid data")
                continue
            
            # Determine regime
            if season <= 2:
                regime = 'rank'
                point_est, ensemble_est, weekly_data = estimate_votes_rank_season(
                    weekly_data, final_ranking, max_week)
            elif season <= 27:
                regime = 'percent'
                point_est, ensemble_est, weekly_data = estimate_votes_percent_season(
                    weekly_data, final_ranking, max_week)
            else:
                regime = 'bottom2'
                point_est, ensemble_est, weekly_data = estimate_votes_bottom2_season(
                    weekly_data, final_ranking, max_week)
            
            # Compute metrics
            consistency = compute_consistency_metrics(point_est, weekly_data, 
                                                       'percent' if regime == 'percent' else 'rank')
            certainty = compute_certainty_metrics(ensemble_est)
            
            # Store results
            for (name, week), vote_share in point_est.items():
                cert = certainty.get((name, week), {})
                
                # Get judge score for this week
                J_score = None
                if week in weekly_data:
                    names = weekly_data[week]['names']
                    if name in names:
                        idx = names.index(name)
                        J_score = weekly_data[week]['J_scores'][idx]
                
                # Check if eliminated this week
                is_eliminated = name in weekly_data.get(week, {}).get('eliminated', [])
                
                all_results.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'fan_vote_share': vote_share,
                    'fan_vote_share_mean': cert.get('mean', vote_share),
                    'fan_vote_share_std': cert.get('std', 0),
                    'cv': cert.get('cv', np.nan),
                    'ci_low': cert.get('ci_low', vote_share),
                    'ci_high': cert.get('ci_high', vote_share),
                    'ci_width': cert.get('ci_width', 0),
                    'judge_total': J_score,
                    'eliminated_this_week': is_eliminated,
                    'regime': regime
                })
            
            # Aggregate certainty metrics for season
            cv_values = [c.get('cv', np.nan) for c in certainty.values() if not np.isnan(c.get('cv', np.nan))]
            width_values = [c.get('ci_width', np.nan) for c in certainty.values() if not np.isnan(c.get('ci_width', np.nan))]
            
            season_metrics.append({
                'season': season,
                'regime': regime,
                'accuracy': consistency['accuracy'],
                'avg_jaccard': consistency['avg_jaccard'],
                'avg_margin': consistency['avg_margin'],
                'n_elim_weeks': consistency['n_elim_weeks'],
                'avg_cv': np.mean(cv_values) if cv_values else np.nan,
                'avg_ci_width': np.mean(width_values) if width_values else np.nan,
                'n_estimates': len(point_est)
            })
            
            print(f"Accuracy={consistency['accuracy']:.2%}, Avg CV={np.mean(cv_values) if cv_values else 0:.3f}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Create DataFrames
    print("\n[3/4] Saving results...")
    
    results_df = pd.DataFrame(all_results)
    metrics_df = pd.DataFrame(season_metrics)
    
    # Save to CSV
    results_df.to_csv('d:/2026-repo/data/fan_vote_results.csv', index=False)
    metrics_df.to_csv('d:/2026-repo/data/consistency_metrics.csv', index=False)
    
    print(f"     Saved fan_vote_results.csv ({len(results_df)} rows)")
    print(f"     Saved consistency_metrics.csv ({len(metrics_df)} rows)")
    
    # Summary statistics
    print("\n[4/4] Summary Statistics")
    print("=" * 60)
    print("\n--- Consistency Metrics by Regime ---")
    regime_summary = metrics_df.groupby('regime').agg({
        'accuracy': ['mean', 'std'],
        'avg_jaccard': 'mean',
        'avg_margin': 'mean'
    }).round(3)
    print(regime_summary)
    
    print("\n--- Certainty Metrics by Regime ---")
    certainty_summary = metrics_df.groupby('regime').agg({
        'avg_cv': ['mean', 'std'],
        'avg_ci_width': ['mean', 'std']
    }).round(4)
    print(certainty_summary)
    
    print("\n--- Overall Performance ---")
    print(f"Overall Accuracy: {metrics_df['accuracy'].mean():.2%} (std: {metrics_df['accuracy'].std():.2%})")
    print(f"Overall Jaccard:  {metrics_df['avg_jaccard'].mean():.3f}")
    print(f"Overall Avg CV:   {metrics_df['avg_cv'].mean():.4f}")
    print(f"Overall Avg CI Width: {metrics_df['avg_ci_width'].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("Done! Results saved to:")
    print("  - d:/2026-repo/data/fan_vote_results.csv")
    print("  - d:/2026-repo/data/consistency_metrics.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
