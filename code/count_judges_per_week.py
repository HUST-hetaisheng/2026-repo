import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('d:/2026-repo/2026_MCM_Problem_C_Data.csv')

# Initialize results dictionary
results = {}

# Ensure we cover seasons 1 to 34
for season in range(1, 35):
    results[season] = {week: 0 for week in range(1, 12)}

# Iterate through each season and week to find the max number of judges
for season in range(1, 35):
    season_df = df[df['season'] == season]
    
    for week in range(1, 12):
        max_judges = 0
        
        # Iterate through all contestants in this season
        for index, row in season_df.iterrows():
            current_contestant_judges = 0
            for judge in range(1, 5):
                col_name = f'week{week}_judge{judge}_score'
                if col_name in df.columns:
                    val = row[col_name]
                    # Check for valid score
                    try:
                        score = pd.to_numeric(val, errors='coerce')
                        if not np.isnan(score) and score > 0:
                            current_contestant_judges += 1
                    except:
                        pass
            
            if current_contestant_judges > max_judges:
                max_judges = current_contestant_judges
        
        results[season][week] = max_judges

# Create DataFrame for output
output_data = []
for season in range(1, 35):
    row_data = {'Season': season}
    for week in range(1, 12):
        row_data[f'Week {week}'] = results[season][week]
    output_data.append(row_data)

output_df = pd.DataFrame(output_data)

# Print the table
print(output_df.to_string(index=False))

# Save to csv
output_df.to_csv('d:/2026-repo/judge_counts_per_week.csv', index=False)
