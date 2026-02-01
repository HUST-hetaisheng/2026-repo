"""
Task 3 数据合并脚本
====================
合并以下数据源:
1. fan_vote_results_final.csv (基础周数据)
2. crawl_celebrity_detailed_info.csv (只保留 bmi, dance_experience_score)
3. crawl_social_media_weekly_data.csv (d,e,f,g 标准化求和为 social_media_popularity，去掉 h 列)
4. 2026_MCM_Problem_C_Data_Cleaned添加人气后.csv (所有信息)

输出: task3_dataset_full.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# === 文件路径 ===
BASE_DIR = r'E:\比赛\数学建模\2026美赛\comap26\2026-repo\data'

FILE_FAN_VOTE = f'{BASE_DIR}\\fan_vote_results_final.csv'
FILE_CELEB_INFO = f'{BASE_DIR}\\crawl_celebrity_detailed_info.csv'
FILE_SOCIAL_WEEKLY = f'{BASE_DIR}\\crawl_social_media_weekly_data.csv'
FILE_CLEANED = f'{BASE_DIR}\\2026_MCM_Problem_C_Data_Cleaned添加人气后.csv'
OUTPUT_FILE = f'{BASE_DIR}\\task3_dataset_full.csv'


def main():
    print("=" * 60)
    print("Task 3 数据合并脚本")
    print("=" * 60)
    
    # === Step 1: 加载基础数据 (fan_vote_results_final.csv) ===
    print("\n[Step 1] 加载 fan_vote_results_final.csv...")
    df_base = pd.read_csv(FILE_FAN_VOTE)
    print(f"  基础数据: {df_base.shape[0]} 行, {df_base.shape[1]} 列")
    
    # === Step 2: 处理 Celebrity Info (只保留 bmi, dance_experience_score) ===
    print("\n[Step 2] 处理 crawl_celebrity_detailed_info.csv...")
    df_celeb = pd.read_csv(FILE_CELEB_INFO)
    # 只保留关键列
    df_celeb_processed = df_celeb[['celebrity_name', 'season', 'bmi', 'dance_experience_score']].copy()
    print(f"  Celebrity Info: 保留 bmi, dance_experience_score")
    print(f"  处理后: {df_celeb_processed.shape[0]} 行")
    
    # === Step 3: 处理 Social Media Weekly (标准化 d,e,f,g 求和，去掉 h) ===
    print("\n[Step 3] 处理 crawl_social_media_weekly_data.csv...")
    df_social = pd.read_csv(FILE_SOCIAL_WEEKLY)
    
    # d,e,f,g 列: twitter_mentions, instagram_engagement, facebook_shares, youtube_views
    # h 列: sentiment_score (不要)
    metrics_cols = ['twitter_mentions', 'instagram_engagement', 'facebook_shares', 'youtube_views']
    
    # 填充缺失值为 0
    df_social[metrics_cols] = df_social[metrics_cols].fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_social[metrics_cols])
    
    # 求和生成 social_media_popularity
    df_social['social_media_popularity'] = scaled_values.sum(axis=1)
    
    # 只保留需要的列 (去掉原始 d,e,f,g 和 h 列)
    cols_to_keep = ['celebrity_name', 'season', 'week', 'social_media_popularity', 'google_search_volume']
    df_social_processed = df_social[cols_to_keep].copy()
    print(f"  标准化 {metrics_cols} 并求和为 social_media_popularity")
    print(f"  去掉 sentiment_score (h 列)")
    print(f"  处理后: {df_social_processed.shape[0]} 行")
    
    # === Step 4: 加载 Cleaned 数据 (所有信息) ===
    print("\n[Step 4] 加载 2026_MCM_Problem_C_Data_Cleaned添加人气后.csv...")
    df_cleaned = pd.read_csv(FILE_CLEANED)
    print(f"  Cleaned 数据: {df_cleaned.shape[0]} 行, {df_cleaned.shape[1]} 列")
    
    # === Step 5: 合并数据 ===
    print("\n[Step 5] 合并数据...")
    
    # 5.1 合并 Cleaned 数据 (按 season, celebrity_name)
    df_merged = pd.merge(
        df_base, 
        df_cleaned, 
        on=['season', 'celebrity_name'], 
        how='left'
    )
    print(f"  合并 Cleaned 后: {df_merged.shape[0]} 行, {df_merged.shape[1]} 列")
    
    # 5.2 合并 Celebrity Info (按 season, celebrity_name)
    df_merged = pd.merge(
        df_merged, 
        df_celeb_processed, 
        on=['season', 'celebrity_name'], 
        how='left'
    )
    print(f"  合并 Celebrity Info 后: {df_merged.shape[0]} 行, {df_merged.shape[1]} 列")
    
    # 5.3 合并 Social Media Weekly (按 season, week, celebrity_name)
    df_merged = pd.merge(
        df_merged, 
        df_social_processed, 
        on=['season', 'week', 'celebrity_name'], 
        how='left'
    )
    print(f"  合并 Social Media 后: {df_merged.shape[0]} 行, {df_merged.shape[1]} 列")
    
    # === Step 6: 保存结果 ===
    print("\n[Step 6] 保存结果...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  最终数据: {df_merged.shape[0]} 行, {df_merged.shape[1]} 列")
    
    # === Step 7: 数据验证 ===
    print("\n[Step 7] 数据验证...")
    print(f"  列名: {df_merged.columns.tolist()}")
    print(f"\n  缺失值统计:")
    missing = df_merged.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = count / len(df_merged) * 100
            print(f"    {col}: {count} ({pct:.1f}%)")
    else:
        print("    无缺失值")
    
    print("\n" + "=" * 60)
    print("合并完成!")
    print("=" * 60)
    
    return df_merged


if __name__ == "__main__":
    df = main()
