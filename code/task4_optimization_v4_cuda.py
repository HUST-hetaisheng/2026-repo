# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统 V4 - CUDA 加速版

核心设计：
1. λ 是每周动态变化的向量 [λ1, λ2, ..., λn]
2. 每周 λ ∈ {0.2, 0.3, ..., 0.8}，共7种选择
3. 使用 CuPy 进行 GPU 并行计算
4. 总收益 = w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy
5. 所有指标归一化到 [0,1]

Author: MCM 2026
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 CuPy
try:
    import cupy as cp
    from cupyx.scipy.stats import rankdata as cp_rankdata
    HAS_CUDA = True
    print("✓ CUDA 可用，使用 GPU 加速")
except ImportError:
    HAS_CUDA = False
    print("✗ CuPy 未安装，使用 CPU 计算")
    cp = np


# ============================================================================
# 数据加载
# ============================================================================

def interpolate_missing_popularity(df, col_name):
    """使用均值填充人气数据中的0值"""
    df = df.copy()
    for season in df['season'].unique():
        season_mask = df['season'] == season
        season_df = df[season_mask]
        for contestant in season_df['celebrity_name'].unique():
            contestant_mask = season_mask & (df['celebrity_name'] == contestant)
            values = df.loc[contestant_mask, col_name].values
            non_zero_values = values[values > 0]
            if len(non_zero_values) == 0:
                global_mean = df[df[col_name] > 0][col_name].mean()
                df.loc[contestant_mask, col_name] = global_mean
            elif len(non_zero_values) < len(values):
                df.loc[contestant_mask & (df[col_name] == 0), col_name] = non_zero_values.mean()
    return df


def load_data():
    """加载原始数据"""
    df_full = pd.read_csv(r'd:\2026-repo\data\task3_dataset_full.csv')
    df_full = interpolate_missing_popularity(df_full, 'Celebrity_Average_Popularity_Score')
    df_full = interpolate_missing_popularity(df_full, 'ballroom_partner_Average_Popularity_Score')
    
    df_fan = pd.read_csv(r'd:\2026-repo\data\fan_vote_results_final.csv')
    pop_cols = ['season', 'week', 'celebrity_name', 'Celebrity_Average_Popularity_Score', 
                'ballroom_partner_Average_Popularity_Score', 'placement']
    df_fan = df_fan.merge(df_full[pop_cols].drop_duplicates(subset=['season', 'week', 'celebrity_name']),
                          on=['season', 'week', 'celebrity_name'], how='left')
    
    df_trends = pd.read_csv(r'd:\2026-repo\data\time_series_US_20040101-0800_20260202-0337.csv')
    df_trends.columns = ['Time', 'search_volume']
    df_trends['Time'] = pd.to_datetime(df_trends['Time'])
    
    schedule = {
        1: ('2005-06-01', '2005-07-06'), 2: ('2006-01-05', '2006-02-26'),
        3: ('2006-09-12', '2006-11-15'), 4: ('2007-03-19', '2007-05-22'),
        5: ('2007-09-24', '2007-11-27'), 6: ('2008-03-17', '2008-05-20'),
        7: ('2008-09-22', '2008-11-25'), 8: ('2009-03-09', '2009-05-19'),
        9: ('2009-09-21', '2009-11-24'), 10: ('2010-03-22', '2010-05-25'),
        11: ('2010-09-20', '2010-11-23'), 12: ('2011-03-21', '2011-05-24'),
        13: ('2011-09-19', '2011-11-22'), 14: ('2012-03-19', '2012-05-22'),
        15: ('2012-09-24', '2012-11-27'), 16: ('2013-03-18', '2013-05-21'),
        17: ('2013-09-16', '2013-11-26'), 18: ('2014-03-17', '2014-05-20'),
        19: ('2014-09-15', '2014-11-25'), 20: ('2015-03-16', '2015-05-19'),
        21: ('2015-09-14', '2015-11-24'), 22: ('2016-03-21', '2016-05-24'),
        23: ('2016-09-12', '2016-11-22'), 24: ('2017-03-20', '2017-05-23'),
        25: ('2017-09-18', '2017-11-21'), 26: ('2018-04-30', '2018-05-21'),
        27: ('2018-09-24', '2018-11-19'), 28: ('2019-09-16', '2019-11-25'),
        29: ('2020-09-14', '2020-11-23'), 30: ('2021-09-20', '2021-11-22'),
        31: ('2022-09-19', '2022-11-21'), 32: ('2023-09-26', '2023-11-21'),
        33: ('2024-09-17', '2024-11-26'), 34: ('2025-02-25', '2025-05-06'),
    }
    return df_full, df_fan, df_trends, schedule


# ============================================================================
# 预处理赛季数据为 GPU 友好格式
# ============================================================================

def preprocess_season_for_gpu(df_season):
    """将赛季数据预处理为 GPU 友好的数组格式"""
    weeks = sorted(df_season['week'].unique())
    n_weeks = len(weeks)
    contestants = df_season['celebrity_name'].unique().tolist()
    n_contestants = len(contestants)
    
    name_to_idx = {name: i for i, name in enumerate(contestants)}
    
    # 预分配数组
    judge_scores = np.zeros((n_weeks, n_contestants))
    fan_shares = np.zeros((n_weeks, n_contestants))
    celeb_pop = np.zeros(n_contestants)
    partner_pop = np.zeros(n_contestants)
    alive_mask = np.ones((n_weeks, n_contestants), dtype=bool)
    eliminated_idx = np.full(n_weeks, -1, dtype=int)
    week_sizes = np.zeros(n_weeks, dtype=int)
    
    # 计算每周参赛者
    current_alive = set(range(n_contestants))
    
    for i, week in enumerate(weeks):
        week_df = df_season[df_season['week'] == week]
        week_contestants = set()
        
        for _, row in week_df.iterrows():
            idx = name_to_idx[row['celebrity_name']]
            week_contestants.add(idx)
            judge_scores[i, idx] = row['judge_total']
            fan_shares[i, idx] = row['fan_vote_share']
            
            if i == 0:
                celeb_pop[idx] = row.get('Celebrity_Average_Popularity_Score', 0) or 0
                partner_pop[idx] = row.get('ballroom_partner_Average_Popularity_Score', 0) or 0
        
        # 设置存活mask
        for idx in range(n_contestants):
            alive_mask[i, idx] = idx in week_contestants
        
        week_sizes[i] = len(week_contestants)
        
        # 找出被淘汰者
        if i + 1 < n_weeks:
            next_week_df = df_season[df_season['week'] == weeks[i + 1]]
            next_contestants = {name_to_idx[row['celebrity_name']] for _, row in next_week_df.iterrows()}
            eliminated = week_contestants - next_contestants
            if eliminated:
                eliminated_idx[i] = list(eliminated)[0]
    
    return {
        'weeks': weeks,
        'n_weeks': n_weeks,
        'n_contestants': n_contestants,
        'contestants': contestants,
        'name_to_idx': name_to_idx,
        'judge_scores': judge_scores,
        'fan_shares': fan_shares,
        'celeb_pop': celeb_pop,
        'partner_pop': partner_pop,
        'alive_mask': alive_mask,
        'eliminated_idx': eliminated_idx,
        'week_sizes': week_sizes,
    }


# ============================================================================
# GPU 加速的批量计算
# ============================================================================

def batch_compute_metrics_gpu(season_data, all_lambda_vecs, regime='percent'):
    """
    GPU 批量计算所有 λ 组合的指标
    
    Parameters:
    - season_data: 预处理的赛季数据
    - all_lambda_vecs: (n_combinations, n_weeks) 的 λ 矩阵
    - regime: 赛制
    
    Returns:
    - metrics: (n_combinations, 5) 的指标矩阵 [celebrity, dancer, ccvi, satisfaction, discrepancy]
    """
    n_weeks = season_data['n_weeks']
    n_contestants = season_data['n_contestants']
    n_combos = len(all_lambda_vecs)
    
    # 转换到 GPU
    if HAS_CUDA:
        judge_scores = cp.asarray(season_data['judge_scores'])
        fan_shares = cp.asarray(season_data['fan_shares'])
        celeb_pop = cp.asarray(season_data['celeb_pop'])
        partner_pop = cp.asarray(season_data['partner_pop'])
        alive_mask = cp.asarray(season_data['alive_mask'])
        eliminated_idx = cp.asarray(season_data['eliminated_idx'])
        lambda_vecs = cp.asarray(all_lambda_vecs)
        xp = cp
    else:
        judge_scores = season_data['judge_scores']
        fan_shares = season_data['fan_shares']
        celeb_pop = season_data['celeb_pop']
        partner_pop = season_data['partner_pop']
        alive_mask = season_data['alive_mask']
        eliminated_idx = season_data['eliminated_idx']
        lambda_vecs = np.array(all_lambda_vecs)
        xp = np
    
    # 初始化结果数组
    celebrity_benefits = xp.zeros(n_combos)
    dancer_effects = xp.zeros(n_combos)
    discrepancies = xp.zeros(n_combos)
    
    # 模拟每个 λ 组合的存活周数
    survival_weeks = xp.zeros((n_combos, n_contestants))
    
    # 对每个组合进行模拟
    for combo_idx in range(n_combos):
        current_alive = xp.ones(n_contestants, dtype=bool)
        
        for week_idx in range(n_weeks):
            lam = lambda_vecs[combo_idx, week_idx]
            
            # 计算综合得分
            week_alive = alive_mask[week_idx] & current_alive
            n_alive = int(xp.sum(week_alive))
            
            if n_alive <= 1:
                survival_weeks[combo_idx, current_alive] = week_idx + 1
                break
            
            # 归一化评分
            judge_sum = xp.sum(judge_scores[week_idx] * week_alive)
            fan_sum = xp.sum(fan_shares[week_idx] * week_alive)
            
            if judge_sum > 0 and fan_sum > 0:
                judge_norm = (judge_scores[week_idx] * week_alive) / judge_sum
                fan_norm = (fan_shares[week_idx] * week_alive) / fan_sum
                
                if regime == 'percent':
                    combined = lam * judge_norm + (1 - lam) * fan_norm
                    # 找最低分者淘汰
                    combined_masked = xp.where(week_alive, combined, xp.inf)
                    elim_idx = int(xp.argmin(combined_masked))
                else:
                    # 排名制
                    judge_rank = xp.zeros(n_contestants)
                    fan_rank = xp.zeros(n_contestants)
                    
                    alive_indices = xp.where(week_alive)[0]
                    judge_alive = judge_scores[week_idx, alive_indices]
                    fan_alive = fan_shares[week_idx, alive_indices]
                    
                    # 排名（CPU 计算）
                    if HAS_CUDA:
                        judge_alive_cpu = cp.asnumpy(judge_alive)
                        fan_alive_cpu = cp.asnumpy(fan_alive)
                        alive_indices_cpu = cp.asnumpy(alive_indices)
                    else:
                        judge_alive_cpu = judge_alive
                        fan_alive_cpu = fan_alive
                        alive_indices_cpu = alive_indices
                    
                    j_ranks = rankdata(-judge_alive_cpu, method='average')
                    f_ranks = rankdata(-fan_alive_cpu, method='average')
                    
                    for local_idx, global_idx in enumerate(alive_indices_cpu):
                        judge_rank[global_idx] = j_ranks[local_idx]
                        fan_rank[global_idx] = f_ranks[local_idx]
                    
                    combined_rank = lam * judge_rank + (1 - lam) * fan_rank
                    combined_rank_masked = xp.where(week_alive, combined_rank, -xp.inf)
                    elim_idx = int(xp.argmax(combined_rank_masked))
                
                current_alive[elim_idx] = False
            
            # 更新存活周数
            survival_weeks[combo_idx, current_alive] = week_idx + 1
        
        # 计算 Celebrity Benefit 和 Dancer Effect
        celebrity_benefits[combo_idx] = xp.sum(survival_weeks[combo_idx] * celeb_pop)
        dancer_effects[combo_idx] = xp.sum(survival_weeks[combo_idx] * partner_pop)
        
        # 计算 Discrepancy（基于历史淘汰反推）
        total_disc = 0.0
        for week_idx in range(n_weeks):
            elim_idx_hist = eliminated_idx[week_idx]
            if elim_idx_hist < 0:
                continue
            
            week_alive = alive_mask[week_idx]
            n_alive = int(xp.sum(week_alive))
            if n_alive == 0:
                continue
            
            lam = lambda_vecs[combo_idx, week_idx]
            
            # 反推粉丝投票：给定 λ 和淘汰者，计算需要的粉丝排名
            # 简化：使用评委排名和淘汰约束估计粉丝排名
            alive_indices = xp.where(week_alive)[0]
            
            if HAS_CUDA:
                judge_alive = cp.asnumpy(judge_scores[week_idx, alive_indices])
                alive_indices_cpu = cp.asnumpy(alive_indices)
                elim_idx_cpu = int(cp.asnumpy(elim_idx_hist))
            else:
                judge_alive = judge_scores[week_idx, alive_indices]
                alive_indices_cpu = alive_indices
                elim_idx_cpu = int(elim_idx_hist)
            
            # 评委排名
            j_ranks = rankdata(-judge_alive, method='average')
            j_rank_pcts = j_ranks / n_alive
            
            # 反推粉丝排名：确保淘汰者综合排名最差
            # 简化模型：粉丝排名 = 评委排名 + 偏移（确保淘汰者最差）
            f_ranks = j_ranks.copy()
            
            # 找到淘汰者在 alive_indices 中的位置
            elim_local_idx = np.where(alive_indices_cpu == elim_idx_cpu)[0]
            if len(elim_local_idx) > 0:
                elim_local_idx = elim_local_idx[0]
                # 确保淘汰者粉丝排名也是最差的
                f_ranks[elim_local_idx] = n_alive
            
            f_rank_pcts = f_ranks / n_alive
            
            # 周分歧度
            week_disc = np.sum(np.abs(f_rank_pcts - j_rank_pcts)) * n_alive
            total_disc += week_disc
        
        discrepancies[combo_idx] = total_disc
    
    # 转回 CPU
    if HAS_CUDA:
        celebrity_benefits = cp.asnumpy(celebrity_benefits)
        dancer_effects = cp.asnumpy(dancer_effects)
        discrepancies = cp.asnumpy(discrepancies)
        survival_weeks = cp.asnumpy(survival_weeks)
    
    return celebrity_benefits, dancer_effects, discrepancies, survival_weeks


def compute_fan_satisfaction_batch(season_data, all_survival_weeks):
    """批量计算观众满意度"""
    n_combos = len(all_survival_weeks)
    n_contestants = season_data['n_contestants']
    
    # 计算每个选手的平均粉丝排名（固定，不随 λ 变化）
    mean_fan_rank_pcts = np.zeros(n_contestants)
    
    for i in range(season_data['n_weeks']):
        alive = season_data['alive_mask'][i]
        n_alive = alive.sum()
        if n_alive == 0:
            continue
        
        fan_alive = season_data['fan_shares'][i, alive]
        f_ranks = rankdata(-fan_alive, method='average')
        f_rank_pcts = f_ranks / n_alive
        
        alive_indices = np.where(alive)[0]
        for local_idx, global_idx in enumerate(alive_indices):
            mean_fan_rank_pcts[global_idx] += f_rank_pcts[local_idx]
    
    # 归一化
    appearances = season_data['alive_mask'].sum(axis=0)
    appearances[appearances == 0] = 1
    mean_fan_rank_pcts /= appearances
    
    # 计算每个组合的满意度
    satisfactions = np.zeros(n_combos)
    
    for combo_idx in range(n_combos):
        surv = all_survival_weeks[combo_idx]
        # 存活周数 → 模拟名次
        placement = rankdata(-surv, method='average')
        placement_pct = placement / n_contestants
        
        # Spearman 相关
        corr, _ = spearmanr(mean_fan_rank_pcts, placement_pct)
        satisfactions[combo_idx] = corr if not np.isnan(corr) else 0
    
    return satisfactions


def compute_ccvi(df_trends, schedule, season):
    """CCVI = 0.3*P + 0.5*I + 0.2*D"""
    if season not in schedule:
        return 0
    start, end = schedule[season]
    start_dt = pd.to_datetime(start) - pd.DateOffset(weeks=1)
    end_dt = pd.to_datetime(end) + pd.DateOffset(weeks=1)
    mask = (df_trends['Time'] >= start_dt) & (df_trends['Time'] <= end_dt)
    season_data = df_trends[mask]['search_volume'].values
    if len(season_data) == 0:
        return 0
    P = np.max(season_data)
    I = np.sum(season_data)
    D = np.max(np.diff(season_data)) if len(season_data) > 1 else 0
    return 0.3 * P + 0.5 * I + 0.2 * max(0, D)


# ============================================================================
# 标准化和收益计算
# ============================================================================

def zscore_normalize(values):
    values = np.array(values, dtype=float)
    mean, std = np.nanmean(values), np.nanstd(values)
    return np.zeros_like(values) if std == 0 or np.isnan(std) else (values - mean) / std


def minmax_normalize(values):
    values = np.array(values, dtype=float)
    min_val, max_val = np.nanmin(values), np.nanmax(values)
    return np.ones_like(values) * 0.5 if max_val == min_val else (values - min_val) / (max_val - min_val)


def compute_total_benefits_batch(celebrities, dancers, ccvi, satisfactions, discrepancies, 
                                  scale_params, weights):
    """
    批量计算总收益
    总收益 = w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy
    """
    w1, w2, w3, w4, w5 = weights
    
    def normalize_batch(vals, col):
        mean, std = scale_params[col]['raw_mean'], scale_params[col]['raw_std']
        z_min, z_max = scale_params[col]['z_min'], scale_params[col]['z_max']
        if std == 0:
            return np.ones_like(vals) * 0.5
        z = (vals - mean) / std
        if z_max == z_min:
            return np.ones_like(vals) * 0.5
        return np.clip((z - z_min) / (z_max - z_min), 0, 1)
    
    celeb_norm = normalize_batch(celebrities, 'celebrity')
    dancer_norm = normalize_batch(dancers, 'dancer')
    ccvi_norm = normalize_batch(np.full_like(celebrities, ccvi), 'ccvi')
    satisfaction_norm = normalize_batch(satisfactions, 'satisfaction')
    discrepancy_norm = normalize_batch(discrepancies, 'discrepancy')
    
    # Discrepancy 取负值
    total = w1 * celeb_norm + w2 * dancer_norm + w3 * ccvi_norm + w4 * satisfaction_norm - w5 * discrepancy_norm
    
    return total, celeb_norm, dancer_norm, ccvi_norm, satisfaction_norm, discrepancy_norm


# ============================================================================
# 预热阶段
# ============================================================================

def warmup_phase(df_fan, df_trends, schedule):
    """预热阶段：获取标准化参数"""
    print("=" * 60)
    print("预热阶段：用 λ=[0.5, ...] 计算各赛季原始指标")
    print("=" * 60)
    
    all_metrics = []
    seasons = sorted(df_fan['season'].unique())
    
    for season in seasons:
        df_season = df_fan[df_fan['season'] == season].copy()
        df_season['Celebrity_Average_Popularity_Score'] = df_season['Celebrity_Average_Popularity_Score'].fillna(0)
        df_season['ballroom_partner_Average_Popularity_Score'] = df_season['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        regime = df_season['regime'].iloc[0] if 'regime' in df_season.columns else 'percent'
        season_data = preprocess_season_for_gpu(df_season)
        n_weeks = season_data['n_weeks']
        
        # λ = 0.5
        lambda_vec = np.array([[0.5] * n_weeks])
        
        celebrities, dancers, discrepancies, survival_weeks = batch_compute_metrics_gpu(
            season_data, lambda_vec, regime)
        satisfactions = compute_fan_satisfaction_batch(season_data, survival_weeks)
        ccvi = compute_ccvi(df_trends, schedule, season)
        
        all_metrics.append({
            'season': season,
            'celebrity': celebrities[0],
            'dancer': dancers[0],
            'ccvi': ccvi,
            'satisfaction': satisfactions[0],
            'discrepancy': discrepancies[0]
        })
        print(f"  Season {season}: celebrity={celebrities[0]:.1f}, discrepancy={discrepancies[0]:.2f}")
    
    metrics_df = pd.DataFrame(all_metrics)
    
    scale_params = {}
    for col in ['celebrity', 'dancer', 'ccvi', 'satisfaction', 'discrepancy']:
        raw_values = metrics_df[col].values
        z_values = zscore_normalize(raw_values)
        scale_params[col] = {
            'raw_mean': np.nanmean(raw_values), 
            'raw_std': np.nanstd(raw_values),
            'z_min': np.nanmin(z_values), 
            'z_max': np.nanmax(z_values)
        }
    
    return metrics_df, scale_params


# ============================================================================
# GPU 加速的网格搜索
# ============================================================================

def search_optimal_lambda_gpu(df_season, df_trends, schedule, season, scale_params, weights, regime='percent'):
    """GPU 加速的网格搜索"""
    season_data = preprocess_season_for_gpu(df_season)
    n_weeks = season_data['n_weeks']
    
    # λ 选项
    lambda_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_options = len(lambda_options)
    total_combinations = n_options ** n_weeks
    
    # 生成所有 λ 组合
    all_lambda_vecs = np.array(list(product(lambda_options, repeat=n_weeks)))
    
    # 批量计算
    ccvi = compute_ccvi(df_trends, schedule, season)
    
    # 分批处理（避免 GPU 内存溢出）
    batch_size = min(10000, total_combinations)
    n_batches = (total_combinations + batch_size - 1) // batch_size
    
    best_benefit = -np.inf
    best_lambda_vec = None
    best_idx = 0
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_combinations)
        batch_lambdas = all_lambda_vecs[start_idx:end_idx]
        
        # GPU 计算
        celebrities, dancers, discrepancies, survival_weeks = batch_compute_metrics_gpu(
            season_data, batch_lambdas, regime)
        satisfactions = compute_fan_satisfaction_batch(season_data, survival_weeks)
        
        # 计算收益
        benefits, _, _, _, _, _ = compute_total_benefits_batch(
            celebrities, dancers, ccvi, satisfactions, discrepancies, scale_params, weights)
        
        # 找最佳
        batch_best_idx = np.argmax(benefits)
        if benefits[batch_best_idx] > best_benefit:
            best_benefit = benefits[batch_best_idx]
            best_lambda_vec = batch_lambdas[batch_best_idx]
            best_idx = start_idx + batch_best_idx
    
    # 计算最佳组合的详细指标
    best_lambda_vec_2d = np.array([best_lambda_vec])
    celebrities, dancers, discrepancies, survival_weeks = batch_compute_metrics_gpu(
        season_data, best_lambda_vec_2d, regime)
    satisfactions = compute_fan_satisfaction_batch(season_data, survival_weeks)
    
    benefits, celeb_norm, dancer_norm, ccvi_norm, sat_norm, disc_norm = compute_total_benefits_batch(
        celebrities, dancers, ccvi, satisfactions, discrepancies, scale_params, weights)
    
    best_metrics = {
        'celebrity': celebrities[0], 'dancer': dancers[0], 'ccvi': ccvi,
        'satisfaction': satisfactions[0], 'discrepancy': discrepancies[0]
    }
    best_norms = {
        'celebrity_norm': celeb_norm[0], 'dancer_norm': dancer_norm[0], 
        'ccvi_norm': ccvi_norm[0], 'satisfaction_norm': sat_norm[0], 
        'discrepancy_norm': disc_norm[0]
    }
    
    return best_lambda_vec, best_benefit, best_metrics, best_norms, total_combinations


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("Task 4: 动态权重优化系统 V4 - CUDA 加速版")
    print("每周 λ ∈ {0.2, 0.3, ..., 0.8}，Discrepancy 取负值")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    df_full, df_fan, df_trends, schedule = load_data()
    print(f"    粉丝投票数据: {len(df_fan)} 条记录")
    
    # 企业权重
    weights = (0.2, 0.1, 0.3, 0.2, 0.2)
    print(f"\n企业权重: w1=0.2, w2=0.1, w3=0.3, w4=0.2, w5=0.2")
    print("收益公式: w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy")
    
    # 预热阶段
    print("\n[2/4] 预热阶段...")
    metrics_df, scale_params = warmup_phase(df_fan, df_trends, schedule)
    
    # 搜索阶段
    print("\n[3/4] 搜索阶段：GPU 加速寻找各赛季最优 λ 向量")
    print("=" * 60)
    
    results = []
    seasons = sorted(df_fan['season'].unique())
    
    for season in seasons:
        df_season = df_fan[df_fan['season'] == season].copy()
        df_season['Celebrity_Average_Popularity_Score'] = df_season['Celebrity_Average_Popularity_Score'].fillna(0)
        df_season['ballroom_partner_Average_Popularity_Score'] = df_season['ballroom_partner_Average_Popularity_Score'].fillna(0)
        
        n_weeks = df_season['week'].nunique()
        regime = df_season['regime'].iloc[0] if 'regime' in df_season.columns else 'percent'
        
        print(f"\n  Season {season} ({regime}, {n_weeks}周)...")
        
        best_lambda_vec, best_benefit, best_metrics, best_norms, total_combos = search_optimal_lambda_gpu(
            df_season, df_trends, schedule, season, scale_params, weights, regime)
        
        # 基准
        baseline_vec = np.array([[0.5] * n_weeks])
        season_data = preprocess_season_for_gpu(df_season)
        base_celeb, base_dancer, base_disc, base_surv = batch_compute_metrics_gpu(season_data, baseline_vec, regime)
        base_sat = compute_fan_satisfaction_batch(season_data, base_surv)
        ccvi = compute_ccvi(df_trends, schedule, season)
        base_benefits, _, _, _, _, _ = compute_total_benefits_batch(
            base_celeb, base_dancer, ccvi, base_sat, base_disc, scale_params, weights)
        baseline_benefit = base_benefits[0]
        
        improvement = best_benefit - baseline_benefit
        
        # 实时打印最佳 λ
        lambda_str = '[' + ', '.join(f'{l:.1f}' for l in best_lambda_vec) + ']'
        print(f"    组合数: {total_combos}")
        print(f"    最佳 λ: {lambda_str}")
        print(f"    收益: {best_benefit:.4f} (基准: {baseline_benefit:.4f}, 改进: {improvement:+.4f})")
        print(f"    Discrepancy: {best_metrics['discrepancy']:.2f} (norm: {best_norms['discrepancy_norm']:.3f})")
        
        results.append({
            'season': season, 'regime': regime, 'n_weeks': n_weeks,
            'total_combinations': total_combos,
            'optimal_lambda_vec': list(best_lambda_vec),
            'optimal_lambda_mean': np.mean(best_lambda_vec),
            'total_benefit': best_benefit,
            'baseline_benefit': baseline_benefit,
            'improvement': improvement,
            **{f'{k}_raw': v for k, v in best_metrics.items()},
            **best_norms
        })
    
    # 保存结果
    print("\n[4/4] 保存结果...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(r'd:\2026-repo\data\task4_optimal_results_v4_cuda.csv', index=False)
    
    # 统计
    print("\n" + "=" * 60)
    print("优化结果统计")
    print("=" * 60)
    print(f"\n平均最优λ: {results_df['optimal_lambda_mean'].mean():.3f}")
    print(f"平均总收益: {results_df['total_benefit'].mean():.4f}")
    print(f"平均改进: {results_df['improvement'].mean():.4f}")
    print(f"改进 > 0 的赛季数: {(results_df['improvement'] > 0).sum()} / {len(results_df)}")
    
    print("\n按赛制分组：")
    print(results_df.groupby('regime')[['optimal_lambda_mean', 'total_benefit', 'improvement']].mean().round(3))
    
    print(f"\n完成！结果保存至 data/task4_optimal_results_v4_cuda.csv")


if __name__ == "__main__":
    main()
