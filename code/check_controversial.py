"""检查争议选手的实际数据"""
import pandas as pd
import numpy as np
from scipy.stats import rankdata

df = pd.read_csv('d:/2026-repo/data/fan_vote_results_final.csv')

CONTROVERSIAL = {
    'Jerry Rice': 2,
    'Billy Ray Cyrus': 4,
    'Bristol Palin': 11,
    'Bobby Bones': 27,
}

print('='*70)
print('争议选手每周详细数据')
print('='*70)

for name, season in CONTROVERSIAL.items():
    subset = df[(df['celebrity_name'] == name) & (df['season'] == season)]
    if len(subset) == 0:
        print(f'{name}: 未找到数据')
        continue
    
    print(f'\n{name} (Season {season}):')
    print(f'  周数: {len(subset)}')
    print(f'  平均裁判分: {subset["judge_total"].mean():.1f}')
    print(f'  平均粉丝份额: {subset["fan_vote_share"].mean():.2%}')
    
    judge_last_count = 0
    fan_first_count = 0
    
    # 检查每周的排名
    for _, row in subset.iterrows():
        week = row['week']
        week_data = df[(df['season'] == season) & (df['week'] == week)]
        n = len(week_data)
        
        # 计算rank
        judge_scores = week_data['judge_total'].values
        fan_shares = week_data['fan_vote_share'].values
        names = week_data['celebrity_name'].values
        
        judge_rank = rankdata(-judge_scores, method='min')
        fan_rank = rankdata(-fan_shares, method='min')
        
        idx = list(names).index(name)
        jr = int(judge_rank[idx])
        fr = int(fan_rank[idx])
        
        if jr == n:
            judge_last_count += 1
        if fr == 1:
            fan_first_count += 1
            
        print(f'  Week {week}: Judge={row["judge_total"]:.0f} (rank {jr}/{n}), Fan={row["fan_vote_share"]:.1%} (rank {fr}/{n})', end='')
        if jr == n:
            print(' *** JUDGE LAST ***', end='')
        if fr == 1:
            print(' *** FAN FIRST ***', end='')
        print()
    
    print(f'  ---')
    print(f'  裁判最后一名次数: {judge_last_count}')
    print(f'  粉丝第一名次数: {fan_first_count}')
