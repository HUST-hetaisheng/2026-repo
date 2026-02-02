import pandas as pd

def check_zeros(file_path, columns, group_col='season'):
    print(f"Checking zeros in {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    stats = []
    
    unique_groups = sorted(df[group_col].unique())
    
    for group in unique_groups:
        subset = df[df[group_col] == group]
        row = {group_col: group, 'count': len(subset)}
        
        has_missing = False
        avg_missing_rate = 0
        
        for col in columns:
            if col in df.columns:
                zero_count = (subset[col] == 0).sum()
                nan_count = subset[col].isna().sum()
                missing = zero_count + nan_count
                pct = (missing / len(subset)) * 100
                row[f'{col}_missing%'] = round(pct, 1)
                avg_missing_rate += pct
            else:
                row[f'{col}_missing%'] = 100.0
                avg_missing_rate += 100.0
        
        row['avg_missing%'] = round(avg_missing_rate / len(columns), 1)
        stats.append(row)
        
    res_df = pd.DataFrame(stats)
    # Sort by completeness (avg_missing% ascending)
    res_df_sorted = res_df.sort_values('avg_missing%')
    
    # Print result
    print(res_df_sorted.to_string())
    return res_df_sorted

print("=== Checking Celebrity Average Popularity Scores (Task 3 Data) ===")
check_zeros(r'd:\2026-repo\data\task3_dataset_full_zscored.csv', 
            ['Celebrity_Average_Popularity_Score', 'ballroom_partner_Average_Popularity_Score'])

print("\n=== Checking Raw Social Media Data (Crawl Weekly) ===")
check_zeros(r'd:\2026-repo\data\crawl_social_media_weekly_data.csv',
            ['twitter_mentions', 'instagram_engagement', 'facebook_shares', 'youtube_views'])

print("\n=== Checking Pre-show Info (Detailed Info) ===")
check_zeros(r'd:\2026-repo\data\crawl_celebrity_detailed_info.csv',
            ['pre_show_social_media_followers_k', 'pre_show_google_search_index'])
