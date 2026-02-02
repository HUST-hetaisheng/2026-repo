import pandas as pd
import numpy as np

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
fan_votes_df = pd.read_csv('E:\比赛\数学建模\2026美赛\comap26\2026-repo\data\fan_vote_results_final.csv')

# Run the simulation
prob_sim_df = run_probabilistic_simulation(fan_votes_df)

# Targets for the response
target_names = ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones", "Nelly", "Iman Shumpert"]
final_view = prob_sim_df[prob_sim_df['celebrity_name'].isin(target_names)].copy()

# Print summary for the last week each target appeared in
summary = final_view.groupby(['celebrity_name', 'season']).tail(1)
print("Final Cumulative Survival Probability for Targets:")
print(summary[['celebrity_name', 'season', 'week', 'cum_p_rank', 'cum_p_pct']])

# Save to CSV
final_view.to_csv('probabilistic_survival_results.csv', index=False)
