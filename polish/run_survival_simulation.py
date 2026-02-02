import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import traceback

print("Starting simulation script...", flush=True)

try:
    # Set font
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except Exception as e:
        print(f"Warning: Could not set font family. {e}")

    def sigmoid(x, k=1.5):
        return 1 / (1 + np.exp(-k * x))

    def run_probabilistic_simulation(df, floor=0.05, k=1.5):
        all_results = []
        
        # Process each season
        for season, s_group in df.groupby('season'):
            weeks = sorted(s_group['week'].unique())
            
            for week in weeks:
                w_data = s_group[s_group['week'] == week].copy()
                
                # Check if anyone was eliminated this week in reality
                num_eliminated = w_data['eliminated_this_week'].sum()
                
                # 1. Calculate Scores
                # Rank Method
                w_data['j_pts'] = w_data['judge_total'].rank(method='average', ascending=True)
                w_data['f_pts'] = w_data['fan_vote_share'].rank(method='average', ascending=True)
                w_data['score_rank'] = w_data['j_pts'] + w_data['f_pts']
                
                # Percentage Method
                sum_j = w_data['judge_total'].sum()
                w_data['j_pct'] = w_data['judge_total'] / (sum_j if sum_j > 0 else 1)
                w_data['score_pct'] = w_data['j_pct'] + w_data['fan_vote_share']
                
                # 2. Convert to Z-scores (Safety Margin)
                # Higher Z is better
                std_rank = w_data['score_rank'].std()
                std_pct = w_data['score_pct'].std()
                
                w_data['z_rank'] = (w_data['score_rank'] - w_data['score_rank'].mean()) / (std_rank if std_rank > 0 else 1)
                w_data['z_pct'] = (w_data['score_pct'] - w_data['score_pct'].mean()) / (std_pct if std_pct > 0 else 1)
                
                # 3. Calculate Survival Probabilities
                if num_eliminated == 0:
                    # Everyone is safe
                    w_data['p_surv_rank'] = 1.0
                    w_data['p_surv_pct'] = 1.0
                else:
                    # Apply sigmoid + floor
                    w_data['p_surv_rank'] = floor + (1 - floor) * sigmoid(w_data['z_rank'], k=k)
                    w_data['p_surv_pct'] = floor + (1 - floor) * sigmoid(w_data['z_pct'], k=k)
                    
                all_results.append(w_data[['season', 'week', 'celebrity_name', 'p_surv_rank', 'p_surv_pct']])
                
        res_df = pd.concat(all_results)
        
        # 4. Calculate Cumulative Probabilities
        res_df = res_df.sort_values(['celebrity_name', 'season', 'week'])
        res_df['cum_p_rank'] = res_df.groupby(['celebrity_name', 'season'])['p_surv_rank'].cumprod()
        res_df['cum_p_pct'] = res_df.groupby(['celebrity_name', 'season'])['p_surv_pct'].cumprod()
        
        return res_df

    # Load the data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes structure: repo/polish/script.py and repo/data/file.csv
    data_path = os.path.join(script_dir, '..', 'data', 'fan_vote_results_final.csv')

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        exit(1)

    fan_votes_df = pd.read_csv(data_path)

    # Run the simulation
    prob_sim_df = run_probabilistic_simulation(fan_votes_df)

    # Targets for the response
    specific_targets = [
        ("Jerry Rice", 2),
        ("Billy Ray Cyrus", 4),
        ("Bristol Palin", 11),
        ("Bobby Bones", 27),
        ("Nelly", 29),
        ("Iman Shumpert", 30)
    ]
    target_names = [t[0] for t in specific_targets]
    final_view = prob_sim_df[prob_sim_df['celebrity_name'].isin(target_names)].copy()

    # Print summary for the last week each target appeared in
    summary = final_view.groupby(['celebrity_name', 'season']).tail(1)
    print("Final Cumulative Survival Probability for Targets:")
    print(summary[['celebrity_name', 'season', 'week', 'cum_p_rank', 'cum_p_pct']])

    # Save to CSV
    output_csv = os.path.join(script_dir, 'probabilistic_survival_results.csv')
    final_view.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # Plotting
    plt.figure(figsize=(12, 8))

    for i, (name, season) in enumerate(specific_targets):
        data = final_view[(final_view['celebrity_name'] == name) & (final_view['season'] == season)].sort_values('week')
        
        plt.subplot(3, 2, i+1)
        
        # Colors: Rank=#87BBA4, Percent=#E57B7F
        plt.plot(data['week'], data['cum_p_rank'], 'o-', label='Rank Rule', color='#87BBA4', alpha=0.9, linewidth=2)
        plt.plot(data['week'], data['cum_p_pct'], 's--', label='Percent Rule', color='#E57B7F', alpha=0.9, linewidth=2)
        
        plt.title(f"{name} (Season {season})")
        plt.yscale('log') # Log scale
        plt.xlabel('Week')
        plt.ylabel('Cum. Survival P')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if i == 0:
            plt.legend()

    plt.tight_layout()
    output_img = os.path.join(script_dir, 'probabilistic_survival_comparison.png')
    plt.savefig(output_img, dpi=1000)
    print(f"Plot saved to {output_img}")

except Exception as e:
    print("An error occurred during the simulation script execution.", file=sys.stderr)
    traceback.print_exc()
    exit(1)
