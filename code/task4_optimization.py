# -*- coding: utf-8 -*-
"""
Task 4: 动态权重优化系统
目标：寻找每周最优评委权重 λ，最大化综合效益

效益函数 = w1*Celebrity_Benefit + w2*Dancer_Effect + w3*CCVI + w4*Fan_Satisfaction - w5*Discrepancy

Author: MCM 2026
"""

import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 阶段一：数据加载与预处理
# ============================================================================

def load_data():
    """加载所有需要的数据"""
    # 主数据集
    df_main = pd.read_csv(r'd:\2026-repo\data\task3_dataset_full_zscored.csv')
    
    # 原始数据 (用于获取存活周数等)
    df_raw = pd.read_csv(r'd:\2026-repo\data\2026_MCM_Problem_C_Data_Cleaned.csv')
    
    # 粉丝投票数据
    df_fan = pd.read_csv(r'd:\2026-repo\data\fan_vote_results_final.csv')
    
    # Google Trends 时间序列
    df_trends = pd.read_csv(r'd:\2026-repo\data\time_series_US_20040101-0800_20260202-0337.csv')
    df_trends.columns = ['Time', 'search_volume']
    df_trends['Time'] = pd.to_datetime(df_trends['Time'])
    
    # 赛程表 - 手动定义（因为原文件编码问题）
    schedule = {
        1: ('2005-06-01', '2005-07-06'),
        2: ('2006-01-05', '2006-02-26'),
        3: ('2006-09-12', '2006-11-15'),
        4: ('2007-03-19', '2007-05-22'),
        5: ('2007-09-24', '2007-11-27'),
        6: ('2008-03-17', '2008-05-20'),
        7: ('2008-09-22', '2008-11-25'),
        8: ('2009-03-09', '2009-05-19'),
        9: ('2009-09-21', '2009-11-24'),
        10: ('2010-03-22', '2010-05-25'),
        11: ('2010-09-20', '2010-11-23'),
        12: ('2011-03-21', '2011-05-24'),
        13: ('2011-09-19', '2011-11-22'),
        14: ('2012-03-19', '2012-05-22'),
        15: ('2012-09-24', '2012-11-27'),
        16: ('2013-03-18', '2013-05-21'),
        17: ('2013-09-16', '2013-11-26'),
        18: ('2014-03-17', '2014-05-20'),
        19: ('2014-09-15', '2014-11-25'),
        20: ('2015-03-16', '2015-05-19'),
        21: ('2015-09-14', '2015-11-24'),
        22: ('2016-03-21', '2016-05-24'),
        23: ('2016-09-12', '2016-11-22'),
        24: ('2017-03-20', '2017-05-23'),
        25: ('2017-09-18', '2017-11-21'),
        26: ('2018-04-30', '2018-05-21'),
        27: ('2018-09-24', '2018-11-19'),
        28: ('2019-09-16', '2019-11-25'),
        29: ('2020-09-14', '2020-11-23'),
        30: ('2021-09-20', '2021-11-22'),
        31: ('2022-09-19', '2022-11-21'),
        32: ('2023-09-26', '2023-12-05'),
        33: ('2024-09-17', '2024-11-26'),
        34: ('2025-09-16', '2025-11-25'),
    }
    
    return df_main, df_raw, df_fan, df_trends, schedule


def lagrange_interpolate_zeros(series):
    """对Series中的0值进行拉格朗日插值填补"""
    s = series.copy()
    # 将0视为缺失
    s = s.replace(0, np.nan)
    
    # 如果全是NaN，返回原值
    if s.isna().all():
        return series
    
    # 获取非NaN的索引和值
    known_idx = s.dropna().index.tolist()
    known_vals = s.dropna().values
    
    if len(known_idx) < 2:
        # 不足两个点，无法插值，用均值填充
        mean_val = known_vals[0] if len(known_vals) > 0 else 0
        return s.fillna(mean_val)
    
    # 构建插值函数
    try:
        poly = lagrange(range(len(known_idx)), known_vals)
        # 对缺失位置进行插值
        for i, val in enumerate(s):
            if pd.isna(val):
                # 找到最近的两个已知点进行局部插值
                interp_val = np.interp(i, range(len(known_idx)), known_vals)
                s.iloc[i] = interp_val
    except:
        # 插值失败，用线性插值
        s = s.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    return s


def zscore(series):
    """Z-score标准化"""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series - mean
    return (series - mean) / std


# ============================================================================
# 阶段二：特征工程 - 五大效益指标
# ============================================================================

def min_max_normalize_series(series):
    """对Series进行Min-Max归一化到[0,1]"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_celebrity_benefit(df_main):
    """
    1. 名人收益 = (social_media_popularity_norm + google_search_volume_norm) * survived_weeks
    
    改进：使用Min-Max归一化替代Z-score，避免负值
    """
    # 按选手聚合（取每人最后一行，因为那时数据最全）
    celeb_data = df_main.groupby(['season', 'celebrity_name']).agg({
        'social_media_popularity': 'last',
        'google_search_volume': 'last',
        'week': 'max',  # survived_weeks
        'placement': 'first'
    }).reset_index()
    
    celeb_data.rename(columns={'week': 'survived_weeks'}, inplace=True)
    
    # Min-Max归一化到[0,1]（替代原来的z-score输入）
    celeb_data['social_media_norm'] = min_max_normalize_series(celeb_data['social_media_popularity'])
    celeb_data['google_search_norm'] = min_max_normalize_series(celeb_data['google_search_volume'])
    
    # 名人热度 = 两个归一化指标的平均（范围[0,1]）
    celeb_data['celebrity_heat'] = (celeb_data['social_media_norm'] + 
                                     celeb_data['google_search_norm']) / 2
    
    # 名人收益 = 热度 * 存活周数（全为正值）
    celeb_data['celebrity_benefit'] = celeb_data['celebrity_heat'] * celeb_data['survived_weeks']
    
    # 按赛季汇总
    season_celeb_benefit = celeb_data.groupby('season')['celebrity_benefit'].sum().reset_index()
    season_celeb_benefit.columns = ['season', 'total_celebrity_benefit']
    
    return season_celeb_benefit


def compute_dancer_effect(df_main):
    """
    2. 舞者效应 = ballroom_partner热度(插值+MinMax归一化) * 存活周数
    
    改进：使用Min-Max归一化替代Z-score，避免负值
    """
    # 对每个舞者的热度进行处理
    dancer_data = df_main.groupby(['season', 'ballroom_partner']).agg({
        'ballroom_partner_Average_Popularity_Score': 'first',
        'week': 'max'
    }).reset_index()
    
    dancer_data.rename(columns={'week': 'survived_weeks'}, inplace=True)
    
    # 拉格朗日插值填补0值
    dancer_data['partner_popularity_filled'] = lagrange_interpolate_zeros(
        dancer_data['ballroom_partner_Average_Popularity_Score']
    )
    
    # Min-Max归一化到[0,1]（替代Z-score，避免负值）
    dancer_data['partner_popularity_norm'] = min_max_normalize_series(
        dancer_data['partner_popularity_filled']
    )
    
    # 舞者效应 = 归一化热度 * 存活周数（全为正值）
    dancer_data['dancer_effect'] = dancer_data['partner_popularity_norm'] * dancer_data['survived_weeks']
    
    # 按赛季汇总
    season_dancer_effect = dancer_data.groupby('season')['dancer_effect'].sum().reset_index()
    season_dancer_effect.columns = ['season', 'total_dancer_effect']
    
    return season_dancer_effect


def compute_ccvi(df_trends, schedule):
    """
    3. 商业价值综合指数 CCVI (基于PID思想)
    P = Peak (峰值)
    I = Integral (积分/总面积)
    D = Derivative (最大上升斜率)
    
    CCVI = 0.3*P_norm + 0.5*I_norm + 0.2*D_norm
    """
    ccvi_results = []
    
    for season, (start, end) in schedule.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        # 扩展时间窗口：前1周到后1周
        start_ext = start_dt - pd.DateOffset(weeks=1)
        end_ext = end_dt + pd.DateOffset(weeks=1)
        
        # 筛选该赛季时间段的数据
        mask = (df_trends['Time'] >= start_ext) & (df_trends['Time'] <= end_ext)
        season_data = df_trends[mask]['search_volume'].values
        
        if len(season_data) == 0:
            ccvi_results.append({'season': season, 'P': 0, 'I': 0, 'D': 0, 'CCVI': 0})
            continue
        
        # P: 峰值
        P = np.max(season_data)
        
        # I: 积分（曲线下面积）
        I = np.sum(season_data)
        
        # D: 最大上升斜率
        if len(season_data) > 1:
            diffs = np.diff(season_data)
            D = np.max(diffs) if len(diffs) > 0 else 0
        else:
            D = 0
        
        ccvi_results.append({'season': season, 'P': P, 'I': I, 'D': D})
    
    ccvi_df = pd.DataFrame(ccvi_results)
    
    # 归一化到0-100
    for col in ['P', 'I', 'D']:
        max_val = ccvi_df[col].max()
        if max_val > 0:
            ccvi_df[f'{col}_norm'] = ccvi_df[col] / max_val * 100
        else:
            ccvi_df[f'{col}_norm'] = 0
    
    # CCVI = 加权组合
    ccvi_df['CCVI'] = 0.3 * ccvi_df['P_norm'] + 0.5 * ccvi_df['I_norm'] + 0.2 * ccvi_df['D_norm']
    
    return ccvi_df[['season', 'P', 'I', 'D', 'CCVI']]


def compute_fan_satisfaction(df_main):
    """
    4. 观众满意度 = Spearman相关系数(平均粉丝排名占比, 最终名次占比)
    
    - 每周粉丝投票排名占比 = fan_rank / remaining_contestants
    - 最终名次占比 = placement / initial_contestants
    - 相关系数越高，说明越迎合观众
    """
    satisfaction_results = []
    
    for season, season_df in df_main.groupby('season'):
        # 获取初始参赛人数
        initial_n = season_df['celebrity_name'].nunique()
        
        # 按选手聚合
        celeb_stats = []
        for celeb, celeb_df in season_df.groupby('celebrity_name'):
            # 计算每周粉丝排名（基于fan_vote_share）
            avg_fan_rank_pct = []
            for week, week_df in celeb_df.groupby('week'):
                # 该周剩余人数
                remaining = len(week_df.groupby('celebrity_name'))
                # 计算该选手的粉丝排名（fan_vote_share降序排名）
                week_all = season_df[season_df['week'] == week].copy()
                week_all['fan_rank'] = week_all['fan_vote_share'].rank(ascending=False)
                fan_rank = week_all[week_all['celebrity_name'] == celeb]['fan_rank'].values
                if len(fan_rank) > 0:
                    fan_rank_pct = fan_rank[0] / len(week_all)
                    avg_fan_rank_pct.append(fan_rank_pct)
            
            # 平均粉丝排名占比
            mean_fan_rank_pct = np.mean(avg_fan_rank_pct) if avg_fan_rank_pct else 0.5
            
            # 最终名次占比
            placement = celeb_df['placement'].iloc[0]
            placement_pct = placement / initial_n
            
            celeb_stats.append({
                'celebrity': celeb,
                'mean_fan_rank_pct': mean_fan_rank_pct,
                'placement_pct': placement_pct
            })
        
        stats_df = pd.DataFrame(celeb_stats)
        
        # 计算相关系数
        if len(stats_df) >= 3:
            corr, _ = spearmanr(stats_df['mean_fan_rank_pct'], stats_df['placement_pct'])
            # 相关系数可能是NaN
            corr = corr if not np.isnan(corr) else 0
        else:
            corr = 0
        
        satisfaction_results.append({'season': season, 'fan_satisfaction': corr})
    
    return pd.DataFrame(satisfaction_results)


def compute_discrepancy(df_main):
    """
    5. 分歧度 = Σ_week Σ_contestant |fan_rank_pct - judge_rank_pct|
    
    - 每周每选手: |粉丝排名占比 - 评委排名占比|
    - 按存活人数加权求和得到周分歧度
    - 按总周数求和得到季分歧度
    - 目标是最小化分歧度
    """
    discrepancy_results = []
    
    for season, season_df in df_main.groupby('season'):
        total_discrepancy = 0
        total_weeks = season_df['week'].nunique()
        
        for week, week_df in season_df.groupby('week'):
            n_remaining = len(week_df)
            
            # 计算粉丝排名（fan_vote_share降序）
            week_df = week_df.copy()
            week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False)
            week_df['fan_rank_pct'] = week_df['fan_rank'] / n_remaining
            
            # 计算评委排名（judge_total降序）
            week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False)
            week_df['judge_rank_pct'] = week_df['judge_rank'] / n_remaining
            
            # 分歧度 = |粉丝排名占比 - 评委排名占比|
            week_df['discrepancy'] = np.abs(week_df['fan_rank_pct'] - week_df['judge_rank_pct'])
            
            # 周分歧度（按人数加权）
            week_discrepancy = week_df['discrepancy'].sum()
            total_discrepancy += week_discrepancy
        
        discrepancy_results.append({
            'season': season, 
            'total_discrepancy': total_discrepancy,
            'total_weeks': total_weeks
        })
    
    return pd.DataFrame(discrepancy_results)


# ============================================================================
# 阶段三：综合效益计算
# ============================================================================

def min_max_normalize(series):
    """Min-Max归一化到[0,1]范围"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_total_benefit(df_main, df_trends, schedule, weights=None):
    """
    计算每季总效益
    
    Total = w1*Celebrity + w2*Dancer + w3*CCVI + w4*Satisfaction - w5*Discrepancy
    
    使用Min-Max归一化将所有指标映射到[0,1]范围，确保无负值且可比较
    
    默认权重：w1=0.2, w2=0.1, w3=0.3, w4=0.2, w5=0.2
    """
    if weights is None:
        weights = [0.2, 0.1, 0.3, 0.2, 0.2]
    
    w1, w2, w3, w4, w5 = weights
    
    # 计算各项指标
    celebrity_benefit = compute_celebrity_benefit(df_main)
    dancer_effect = compute_dancer_effect(df_main)
    ccvi = compute_ccvi(df_trends, schedule)
    fan_satisfaction = compute_fan_satisfaction(df_main)
    discrepancy = compute_discrepancy(df_main)
    
    # 合并所有指标
    result = celebrity_benefit.merge(dancer_effect, on='season')
    result = result.merge(ccvi[['season', 'CCVI']], on='season')
    result = result.merge(fan_satisfaction, on='season')
    result = result.merge(discrepancy[['season', 'total_discrepancy']], on='season')
    
    # Min-Max归一化各指标到[0,1]范围（解决数据类型不同和负值问题）
    result['celebrity_norm'] = min_max_normalize(result['total_celebrity_benefit'])
    result['dancer_norm'] = min_max_normalize(result['total_dancer_effect'])
    result['ccvi_norm'] = min_max_normalize(result['CCVI'])
    result['satisfaction_norm'] = min_max_normalize(result['fan_satisfaction'])
    # 分歧度需要反转：原值越大越差，归一化后越小越好
    result['discrepancy_norm'] = 1 - min_max_normalize(result['total_discrepancy'])
    
    # 计算总效益（所有项都是正向的，范围都在[0,1]）
    result['total_benefit'] = (
        w1 * result['celebrity_norm'] +
        w2 * result['dancer_norm'] +
        w3 * result['ccvi_norm'] +
        w4 * result['satisfaction_norm'] +
        w5 * result['discrepancy_norm']  # 已反转，越大越好
    )
    
    return result


# ============================================================================
# 阶段四：动态权重优化
# ============================================================================

def simulate_season_with_lambda(df_season, lambda_vec):
    """
    给定一个赛季的数据和每周评委权重λ向量，模拟该赛季的结果
    
    综合得分 = λ * judge_score_pct + (1-λ) * fan_vote_pct
    
    返回模拟后的各项指标
    """
    weeks = sorted(df_season['week'].unique())
    n_weeks = len(weeks)
    
    # 如果λ向量长度不够，用最后一个值填充
    if len(lambda_vec) < n_weeks:
        lambda_vec = list(lambda_vec) + [lambda_vec[-1]] * (n_weeks - len(lambda_vec))
    
    simulated_results = []
    eliminated = set()
    
    for i, week in enumerate(weeks):
        lam = lambda_vec[i] if i < len(lambda_vec) else 0.5
        
        week_df = df_season[df_season['week'] == week].copy()
        # 排除已淘汰选手
        week_df = week_df[~week_df['celebrity_name'].isin(eliminated)]
        
        if len(week_df) == 0:
            continue
        
        # 计算综合得分
        # judge_score_pct
        total_judge = week_df['judge_total'].sum()
        if total_judge > 0:
            week_df['judge_pct'] = week_df['judge_total'] / total_judge
        else:
            week_df['judge_pct'] = 1 / len(week_df)
        
        # fan_vote_pct (已经有fan_vote_share)
        week_df['combined_score'] = lam * week_df['judge_pct'] + (1 - lam) * week_df['fan_vote_share']
        
        # 按综合得分排名
        week_df['combined_rank'] = week_df['combined_score'].rank(ascending=True)
        
        # 最低分者被淘汰
        min_score_idx = week_df['combined_score'].idxmin()
        eliminated_celeb = week_df.loc[min_score_idx, 'celebrity_name']
        eliminated.add(eliminated_celeb)
        
        # 记录结果
        for _, row in week_df.iterrows():
            simulated_results.append({
                'week': week,
                'celebrity_name': row['celebrity_name'],
                'combined_score': row['combined_score'],
                'combined_rank': row['combined_rank'],
                'eliminated': row['celebrity_name'] == eliminated_celeb
            })
    
    return pd.DataFrame(simulated_results)


def optimize_lambda_for_season(df_season, target_weights=[0.2, 0.1, 0.3, 0.2, 0.2]):
    """
    使用差分进化算法寻找最优λ向量
    
    目标：最大化总效益
    """
    n_weeks = df_season['week'].nunique()
    
    def objective(lambda_vec):
        """目标函数（最小化负效益）"""
        # 模拟赛季
        sim_results = simulate_season_with_lambda(df_season, lambda_vec)
        
        if len(sim_results) == 0:
            return 1e10
        
        # 计算分歧度
        total_disc = 0
        for week in sim_results['week'].unique():
            week_data = sim_results[sim_results['week'] == week]
            n = len(week_data)
            if n > 0:
                # 简化：用combined_rank的方差作为分歧度代理
                disc = week_data['combined_rank'].std() / n if n > 1 else 0
                total_disc += disc
        
        # 计算存活周数（非淘汰选手的平均周数）
        survival = sim_results.groupby('celebrity_name')['week'].max().mean()
        
        # 简化的效益函数
        benefit = survival - 0.5 * total_disc
        
        return -benefit  # 最小化负效益 = 最大化效益
    
    # 边界：每周λ在[0, 1]之间
    bounds = [(0.1, 0.9)] * n_weeks
    
    # 差分进化优化
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=100,
        seed=42,
        workers=1,
        disp=False
    )
    
    return result.x, -result.fun


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("Task 4: 动态权重优化系统")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/6] 加载数据...")
    df_main, df_raw, df_fan, df_trends, schedule = load_data()
    print(f"    主数据集: {len(df_main)} 条记录")
    print(f"    覆盖赛季: {df_main['season'].min()} - {df_main['season'].max()}")
    
    # 计算各项指标
    print("\n[2/6] 计算名人收益...")
    celebrity_benefit = compute_celebrity_benefit(df_main)
    print(f"    完成 {len(celebrity_benefit)} 个赛季")
    
    print("\n[3/6] 计算舞者效应...")
    dancer_effect = compute_dancer_effect(df_main)
    print(f"    完成 {len(dancer_effect)} 个赛季")
    
    print("\n[4/6] 计算CCVI商业价值指数...")
    ccvi = compute_ccvi(df_trends, schedule)
    print(ccvi[['season', 'CCVI']].head(10).to_string())
    
    print("\n[5/6] 计算观众满意度...")
    fan_satisfaction = compute_fan_satisfaction(df_main)
    print(f"    平均满意度相关系数: {fan_satisfaction['fan_satisfaction'].mean():.4f}")
    
    print("\n[6/6] 计算分歧度...")
    discrepancy = compute_discrepancy(df_main)
    print(f"    平均分歧度: {discrepancy['total_discrepancy'].mean():.4f}")
    
    # 计算总效益
    print("\n" + "=" * 60)
    print("综合效益计算 (Min-Max归一化)")
    print("=" * 60)
    
    total_benefit = compute_total_benefit(df_main, df_trends, schedule)
    
    # 显示结果（使用归一化后的列）
    display_cols = ['season', 'ccvi_norm', 'satisfaction_norm', 'discrepancy_norm', 'total_benefit']
    print("\n各赛季综合效益排名（Top 10）:")
    print(total_benefit.sort_values('total_benefit', ascending=False)[display_cols].head(10).to_string())
    
    print("\n各赛季综合效益排名（Bottom 5）:")
    print(total_benefit.sort_values('total_benefit', ascending=True)[display_cols].head(5).to_string())
    
    # 保存结果
    output_path = r'd:\2026-repo\data\task4_season_benefits.csv'
    total_benefit.to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")
    
    # 动态权重优化示例（以第27季为例）
    print("\n" + "=" * 60)
    print("动态权重优化示例：第27季 (Bobby Bones)")
    print("=" * 60)
    
    df_s27 = df_main[df_main['season'] == 27].copy()
    print(f"第27季数据: {len(df_s27)} 条记录, {df_s27['week'].nunique()} 周")
    
    print("\n正在寻找最优评委权重λ...")
    optimal_lambda, optimal_benefit = optimize_lambda_for_season(df_s27)
    
    print(f"\n最优λ向量 (每周评委权重):")
    for i, lam in enumerate(optimal_lambda):
        print(f"    Week {i+1}: λ = {lam:.4f}")
    print(f"\n最优效益值: {optimal_benefit:.4f}")
    
    # 保存优化结果
    lambda_df = pd.DataFrame({
        'week': range(1, len(optimal_lambda) + 1),
        'optimal_lambda': optimal_lambda
    })
    lambda_path = r'd:\2026-repo\data\task4_optimal_lambda_s27.csv'
    lambda_df.to_csv(lambda_path, index=False)
    print(f"最优λ已保存至: {lambda_path}")
    
    print("\n" + "=" * 60)
    print("Task 4 完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
