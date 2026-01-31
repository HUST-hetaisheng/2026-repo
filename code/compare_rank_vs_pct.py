"""
MCM 2026 Problem C - Q2: Compare Rank vs Percentage Voting Methods
==================================================================
对每一季同时应用两种计分方法：
1. Rank Method (S1-2实际使用): 评委分数转排名 + 观众投票转排名 → 排名相加
2. Percentage Method (S3-27实际使用): 评委分数百分比 + 观众投票百分比 → 加权平均

分析：
- 两种方法对每周淘汰结果的预测差异
- 哪种方法更倾向于观众投票 (即让评委分数低但人气高的选手存活)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os

# ============================================================
# 数据加载
# ============================================================
data_path = r'd:\2026-repo\data\2026_MCM_Problem_C_Data_Cleaned.csv'
df = pd.read_csv(data_path)

# 获取评委分数列
judge_cols = [c for c in df.columns if 'judge' in c.lower() and 'score' in c.lower()]

# 解析周数
def get_week_number(col):
    """从列名提取周数，如 week1_judge1_score -> 1"""
    import re
    match = re.search(r'week(\d+)', col)
    return int(match.group(1)) if match else None

# 按周组织评委分数列
weeks_cols = defaultdict(list)
for col in judge_cols:
    week = get_week_number(col)
    if week:
        weeks_cols[week].append(col)

# ============================================================
# 核心函数：计算每周的综合得分
# ============================================================

def compute_weekly_judge_total(row, week):
    """计算某选手某周的评委总分"""
    cols = weeks_cols.get(week, [])
    scores = [row[c] for c in cols if pd.notna(row[c]) and row[c] > 0]
    return sum(scores) if scores else 0

def get_contestants_in_week(season_df, week):
    """获取某周还在比赛的选手（评委分数 > 0）"""
    contestants = []
    for idx, row in season_df.iterrows():
        total = compute_weekly_judge_total(row, week)
        if total > 0:
            contestants.append({
                'name': row['celebrity_name'],
                'judge_total': total,
                'placement': row['placement'],  # 最终名次
                'results': row['results']
            })
    return contestants

def rank_method(contestants, fan_votes):
    """
    Rank Method: 
    - 评委分数转排名 (分数高 → 排名小/好)
    - 观众投票转排名 (票数高 → 排名小/好)
    - 最终排名 = 评委排名 + 观众排名 (越小越好)
    
    fan_votes: dict {name: vote_share}
    """
    n = len(contestants)
    
    # 评委排名 (分数高排前面)
    sorted_by_judge = sorted(contestants, key=lambda x: x['judge_total'], reverse=True)
    judge_rank = {c['name']: i+1 for i, c in enumerate(sorted_by_judge)}
    
    # 观众排名 (票数高排前面)
    sorted_by_fan = sorted(contestants, key=lambda x: fan_votes.get(x['name'], 0), reverse=True)
    fan_rank = {c['name']: i+1 for i, c in enumerate(sorted_by_fan)}
    
    # 综合排名
    combined = []
    for c in contestants:
        name = c['name']
        jr = judge_rank[name]
        fr = fan_rank[name]
        combined.append({
            'name': name,
            'judge_rank': jr,
            'fan_rank': fr,
            'combined_rank_score': jr + fr,  # 越小越好
            'judge_total': c['judge_total'],
            'fan_vote': fan_votes.get(name, 0)
        })
    
    # 按综合分排序 (分数低 = 名次好)
    combined.sort(key=lambda x: x['combined_rank_score'])
    
    # 最后一名被淘汰
    return combined

def percentage_method(contestants, fan_votes, fan_weight=0.5):
    """
    Percentage Method:
    - 评委分数百分比 = 个人分数 / 总分数
    - 观众投票百分比 = 个人票数 / 总票数 (直接用 fan_vote_share)
    - 最终得分 = (1 - fan_weight) * 评委百分比 + fan_weight * 观众百分比
    
    fan_votes: dict {name: vote_share}
    """
    total_judge = sum(c['judge_total'] for c in contestants)
    total_fan = sum(fan_votes.get(c['name'], 0) for c in contestants)
    
    combined = []
    for c in contestants:
        name = c['name']
        judge_pct = c['judge_total'] / total_judge if total_judge > 0 else 0
        fan_pct = fan_votes.get(name, 0) / total_fan if total_fan > 0 else 0
        
        # 50-50 加权
        final_score = (1 - fan_weight) * judge_pct + fan_weight * fan_pct
        
        combined.append({
            'name': name,
            'judge_pct': judge_pct,
            'fan_pct': fan_pct,
            'final_score': final_score,
            'judge_total': c['judge_total'],
            'fan_vote': fan_votes.get(name, 0)
        })
    
    # 按最终得分排序 (得分高 = 名次好)
    combined.sort(key=lambda x: x['final_score'], reverse=True)
    
    return combined

# ============================================================
# 模拟观众投票：使用之前模型的结果
# ============================================================
fan_vote_path = r'd:\2026-repo\data\fan_vote_results_final.csv'
if os.path.exists(fan_vote_path):
    df_fan = pd.read_csv(fan_vote_path)
else:
    # 如果没有预测结果，使用均匀分布作为占位符
    df_fan = None
    print("Warning: fan_vote_results_final.csv not found. Using uniform fan votes.")

def get_fan_votes(season, week):
    """获取某季某周的观众投票分布"""
    if df_fan is not None:
        subset = df_fan[(df_fan['season'] == season) & (df_fan['week'] == week)]
        return dict(zip(subset['celebrity_name'], subset['fan_vote_share']))
    else:
        return {}

# ============================================================
# 主分析：对比两种方法
# ============================================================

results = []
season_summary = []

for season in sorted(df['season'].unique()):
    season_df = df[df['season'] == season]
    max_week = 11  # 最多11周
    
    season_results = {
        'season': season,
        'rank_eliminated': [],
        'pct_eliminated': [],
        'actual_eliminated': [],
        'disagreements': 0,
        'fan_favored_by_rank': 0,
        'fan_favored_by_pct': 0
    }
    
    for week in range(1, max_week + 1):
        contestants = get_contestants_in_week(season_df, week)
        if len(contestants) < 2:
            break
            
        fan_votes = get_fan_votes(season, week)
        
        # 如果没有观众投票数据，使用均匀分布
        if not fan_votes:
            for c in contestants:
                fan_votes[c['name']] = 1.0 / len(contestants)
        
        # 应用两种方法
        rank_result = rank_method(contestants, fan_votes)
        pct_result = percentage_method(contestants, fan_votes)
        
        # 获取两种方法预测的淘汰者 (最后一名)
        rank_eliminated = rank_result[-1]['name'] if rank_result else None
        pct_eliminated = pct_result[-1]['name'] if pct_result else None
        
        # 找出实际淘汰者
        actual_eliminated = None
        for c in contestants:
            if 'Eliminated Week' in str(c['results']) and f'Week {week}' in str(c['results']):
                actual_eliminated = c['name']
                break
        
        # 记录结果
        week_result = {
            'season': season,
            'week': week,
            'num_contestants': len(contestants),
            'rank_eliminated': rank_eliminated,
            'pct_eliminated': pct_eliminated,
            'actual_eliminated': actual_eliminated,
            'methods_agree': rank_eliminated == pct_eliminated
        }
        
        # 分析哪种方法更倾向于观众
        if rank_eliminated and pct_eliminated and rank_eliminated != pct_eliminated:
            season_results['disagreements'] += 1
            
            # 找到两个被淘汰者的信息
            rank_elim_info = next((r for r in rank_result if r['name'] == rank_eliminated), None)
            pct_elim_info = next((r for r in pct_result if r['name'] == pct_eliminated), None)
            
            if rank_elim_info and pct_elim_info:
                # Rank方法淘汰的人的观众投票
                rank_elim_fan = rank_elim_info['fan_vote']
                # Pct方法淘汰的人的观众投票
                pct_elim_fan = pct_elim_info['fan_vote']
                
                # 如果 Rank 淘汰的人观众票更高，说明 Rank 更不利于高人气选手
                # 反之，Pct 更不利于高人气选手
                if rank_elim_fan > pct_elim_fan:
                    week_result['fan_favored_by'] = 'Percentage'
                    season_results['fan_favored_by_pct'] += 1
                else:
                    week_result['fan_favored_by'] = 'Rank'
                    season_results['fan_favored_by_rank'] += 1
                    
                week_result['rank_elim_fan_vote'] = rank_elim_fan
                week_result['pct_elim_fan_vote'] = pct_elim_fan
        
        results.append(week_result)
        
        if actual_eliminated:
            season_results['actual_eliminated'].append(actual_eliminated)
        season_results['rank_eliminated'].append(rank_eliminated)
        season_results['pct_eliminated'].append(pct_eliminated)
    
    season_summary.append(season_results)

# ============================================================
# 汇总统计
# ============================================================

df_results = pd.DataFrame(results)
df_summary = pd.DataFrame(season_summary)

# 总体统计
total_weeks = len(df_results)
agreements = df_results['methods_agree'].sum()
disagreements = total_weeks - agreements

print("=" * 60)
print("RANK vs PERCENTAGE METHOD COMPARISON")
print("=" * 60)
print(f"\nTotal weeks analyzed: {total_weeks}")
print(f"Methods agree: {agreements} ({100*agreements/total_weeks:.1f}%)")
print(f"Methods disagree: {disagreements} ({100*disagreements/total_weeks:.1f}%)")

# 哪种方法更倾向于观众
total_fan_favored_rank = df_summary['fan_favored_by_rank'].sum()
total_fan_favored_pct = df_summary['fan_favored_by_pct'].sum()

print(f"\nWhen methods disagree:")
print(f"  Fan votes favored by RANK method: {total_fan_favored_rank} times")
print(f"  Fan votes favored by PERCENTAGE method: {total_fan_favored_pct} times")

if total_fan_favored_pct > total_fan_favored_rank:
    print(f"\n>>> CONCLUSION: PERCENTAGE method tends to FAVOR fan votes more.")
    print(f"    (It keeps high-popularity contestants alive more often)")
elif total_fan_favored_rank > total_fan_favored_pct:
    print(f"\n>>> CONCLUSION: RANK method tends to FAVOR fan votes more.")
    print(f"    (It keeps high-popularity contestants alive more often)")
else:
    print(f"\n>>> CONCLUSION: Both methods have similar bias toward fan votes.")

# 按赛季展示
print("\n" + "=" * 60)
print("BY-SEASON BREAKDOWN")
print("=" * 60)
print(f"{'Season':<8} {'Disagree':<10} {'Fan→Rank':<10} {'Fan→Pct':<10}")
print("-" * 40)
for s in season_summary:
    print(f"{s['season']:<8} {s['disagreements']:<10} {s['fan_favored_by_rank']:<10} {s['fan_favored_by_pct']:<10}")

# 保存详细结果
output_path = r'd:\2026-repo\data\rank_vs_pct_comparison.csv'
df_results.to_csv(output_path, index=False)
print(f"\nDetailed results saved to: {output_path}")

# 保存汇总
summary_path = r'd:\2026-repo\data\rank_vs_pct_summary.csv'
df_summary.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")

# ============================================================
# 深入分析：量化偏好差异
# ============================================================

print("\n" + "=" * 60)
print("QUANTITATIVE ANALYSIS: Fan Vote Influence")
print("=" * 60)

# 对于每个不一致的周次，计算观众投票差异
disagree_weeks = df_results[df_results['methods_agree'] == False].copy()

if 'rank_elim_fan_vote' in disagree_weeks.columns:
    avg_rank_elim_fan = disagree_weeks['rank_elim_fan_vote'].mean()
    avg_pct_elim_fan = disagree_weeks['pct_elim_fan_vote'].mean()
    
    print(f"\nIn weeks where methods disagree:")
    print(f"  Avg fan vote of contestant eliminated by RANK: {avg_rank_elim_fan:.4f}")
    print(f"  Avg fan vote of contestant eliminated by PCT:  {avg_pct_elim_fan:.4f}")
    
    diff = avg_rank_elim_fan - avg_pct_elim_fan
    if diff > 0:
        print(f"\n  → RANK method eliminates contestants with {diff:.4f} HIGHER fan votes on average")
        print(f"  → This means PERCENTAGE method is MORE FAVORABLE to popular contestants")
    else:
        print(f"\n  → PERCENTAGE method eliminates contestants with {-diff:.4f} HIGHER fan votes on average")
        print(f"  → This means RANK method is MORE FAVORABLE to popular contestants")
