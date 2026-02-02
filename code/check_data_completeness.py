import pandas as pd
import numpy as np

def check_completeness(file_path, columns, name):
    print(f"--- Checking {name} ---")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    results = []
    
    # Check if season column exists
    if 'season' not in df.columns:
        print(" 'season' column not found.")
        return

    for season, group in df.groupby('season'):
        season_stats = {'season': season, 'count': len(group)}
        for col in columns:
            if col in df.columns:
                # Count NaNs
                nans = group[col].isna().sum()
                # Count Zeroes (sometimes missing data is 0)
                zeroes = (group[col] == 0).sum()
                
                season_stats[f'{col}_nan'] = nans
                season_stats[f'{col}_zero'] = zeroes
                season_stats[f'{col}_missing_pct'] = (nans + zeroes) / len(group) * 100
            else:
                season_stats[f'{col}_missing_pct'] = 100.0 # Column missing
        results.append(season_stats)

    results_df = pd.DataFrame(results)
    
    # Calculate an overall completeness score (lower missing % is better)
    # We aggregate the missing percentages of the target columns
    missing_cols = [c for c in results_df.columns if 'missing_pct' in c]
    results_df['avg_missing_pct'] = results_df[missing_cols].mean(axis=1)
    
    # Sort by completeness (avg_missing_pct ascending)
    results_df_sorted = results_df.sort_values('avg_missing_pct')
    
    print(results_df_sorted[['season', 'count', 'avg_missing_pct'] + missing_cols].to_string())
    print("\nTop 5 seasons with most complete data:")
    print(results_df_sorted.head(5)[['season', 'avg_missing_pct']])

# Check processed data
check_completeness(
    r'd:\2026-repo\data\task3_dataset_full_zscored.csv', 
    ['social_media_popularity', 'google_search_volume', 'fan_vote_share'], 
    'Processed Dataset'
)

# Check raw social media data
check_completeness(
    r'd:\2026-repo\data\crawl_social_media_weekly_data.csv',
    ['twitter_mentions', 'instagram_engagement', 'facebook_shares', 'youtube_views'],
    'Raw Social Media Data'
)
