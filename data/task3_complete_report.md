# Task 3 Complete Analysis Report
# Impact of Contestant and Partner Characteristics on DWTS Performance

*Generated: 2026-01-31*

---

## Executive Summary

本报告分析了《Dancing with the Stars》(DWTS) 选手和舞伴特征对比赛结果的影响。主要发现：

1. **年龄效应**：年龄对评委打分有显著负向影响（β = -0.34, p < 0.001），年轻选手表现更好
2. **人气效应差异**：名人人气对粉丝投票影响显著（t = 2.85, p < 0.01），但对评委打分无显著影响
3. **舞伴效应**：职业舞者选择解释了约13.7%-15.3%的结果方差，Derek Hough是表现最好的舞伴
4. **评委vs粉丝**：两者中度相关（r = 0.52），但评价标准存在系统性差异

---

## Part 1: Data Overview

### 1.1 Sample Description

| Statistic | Value |
|-----------|-------|
| Total Contestants | 421 |
| Total Weekly Records | 2,777 |
| Unique Professional Dancers | 60 |
| Dancers with 2+ Appearances | 40 |
| Seasons Covered | 1-33 |

### 1.2 Variables Used

**Dependent Variables (因变量):**
- `avg_judge_score`: 平均每周评委总分 (0-40)
- `avg_fan_vote_share`: 平均粉丝投票份额 (0-1)
- `placement`: 最终名次 (1 = 冠军)

**Independent Variables (自变量):**
- `age`: 参赛时年龄
- `age_squared`: 年龄平方项（捕捉非线性）
- `is_us`: 是否美国出生 (1/0)
- `celeb_pop_log`: log(名人社交媒体人气 + 1)
- `partner_pop_log`: log(舞伴社交媒体人气 + 1)
- `partner_seasons_before`: 舞伴历史参赛次数

---

## Part 2: Module A - Celebrity Characteristics Analysis (OLS)

### 2.1 Model Specification

$$Y_i = \beta_0 + \beta_1 \cdot Age_i + \beta_2 \cdot Age_i^2 + \beta_3 \cdot isUS_i + \beta_4 \cdot \log(CelebPop_i + 1) + \beta_5 \cdot \log(PartnerPop_i + 1) + \epsilon_i$$

### 2.2 Results Summary

#### Model 1: Average Judge Score (R² = 0.193)

| Variable | Coefficient | Std Error | t-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
| Intercept | 33.790 | 2.031 | 16.63 | <0.001 | *** |
| age | **-0.343** | 0.089 | -3.84 | <0.001 | *** |
| age_squared | 0.002 | 0.001 | 2.01 | 0.045 | ** |
| is_us | 0.115 | 0.706 | 0.16 | 0.871 | |
| celeb_pop_log | -0.719 | 0.609 | -1.18 | 0.238 | |
| partner_pop_log | 0.861 | 0.612 | 1.41 | 0.161 | |

**Key Insight:** 年龄是评委打分的最强预测因子。每增加1岁，评委分平均下降0.34分。

#### Model 2: Average Fan Vote Share (R² = 0.147)

| Variable | Coefficient | Std Error | t-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
| Intercept | 0.167 | 0.019 | 8.85 | <0.001 | *** |
| age | **-0.002** | 0.001 | -2.20 | 0.029 | ** |
| age_squared | 0.000 | 0.000 | 0.68 | 0.494 | |
| is_us | -0.002 | 0.007 | -0.28 | 0.776 | |
| celeb_pop_log | **0.016** | 0.006 | 2.85 | 0.005 | *** |
| partner_pop_log | **-0.016** | 0.006 | -2.88 | 0.004 | *** |

**Key Insight:** 名人社交媒体人气显著提升粉丝投票份额。每增加1单位log人气，粉丝份额增加1.6%。

#### Model 3: Final Placement (R² = 0.207)

| Variable | Coefficient | Std Error | t-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
| Intercept | 0.559 | 1.396 | 0.40 | 0.689 | |
| age | **0.226** | 0.061 | 3.69 | <0.001 | *** |
| age_squared | -0.001 | 0.001 | -1.83 | 0.068 | * |
| is_us | -0.081 | 0.485 | -0.17 | 0.868 | |
| celeb_pop_log | **-1.094** | 0.418 | -2.62 | 0.009 | *** |
| partner_pop_log | **0.924** | 0.421 | 2.20 | 0.029 | ** |

**Key Insight:** 高人气名人名次更好（负系数 = 更低名次 = 更好表现）。

---

## Part 3: Module B - Professional Dancer Analysis (Mixed Effects)

### 3.1 Model Specification

$$Y_{ij} = \beta_0 + \beta_1 \cdot Age_{ij} + \beta_2 \cdot isUS_{ij} + \beta_3 \cdot CelebPopLog_{ij} + \beta_4 \cdot PartnerExp_j + u_j + \epsilon_{ij}$$

其中 $u_j \sim N(0, \sigma^2_u)$ 是舞者 $j$ 的随机效应。

### 3.2 Mixed Model Results

#### Judge Score Model

| Component | Value | Interpretation |
|-----------|-------|----------------|
| **ICC** | **13.66%** | 13.7%的评委分方差由舞者差异解释 |
| σ²_partner | 3.201 | 舞者间方差 |
| σ²_residual | 20.230 | 残差方差 |

**Fixed Effects:**
| Variable | Coefficient | z-value | p-value | Significance |
|----------|-------------|---------|---------|--------------|
| Intercept | 28.287 | 25.30 | <0.001 | *** |
| age | -0.153 | -8.75 | <0.001 | *** |
| is_us | -0.044 | -0.07 | 0.947 | |
| celeb_pop_log | 0.329 | 1.03 | 0.304 | |
| **partner_seasons_before** | **0.196** | **4.05** | **<0.001** | *** |

**Key Insight:** 舞伴经验显著正向影响评委分。每多参加一季，选手评委分平均提高0.2分。

#### Fan Vote Model

| Component | Value | Interpretation |
|-----------|-------|----------------|
| **ICC** | **15.26%** | 15.3%的粉丝投票方差由舞者差异解释 |
| σ²_partner | 0.0003 | 舞者间方差 |
| σ²_residual | 0.0018 | 残差方差 |

**Fixed Effects:**
| Variable | Coefficient | z-value | p-value | Significance |
|----------|-------------|---------|---------|--------------|
| Intercept | 0.158 | 14.47 | <0.001 | *** |
| age | -0.001 | -7.22 | <0.001 | *** |
| is_us | -0.000 | -0.01 | 0.993 | |
| celeb_pop_log | 0.001 | 0.25 | 0.800 | |
| **partner_seasons_before** | **-0.002** | **-3.34** | **<0.001** | *** |

**Surprising Finding:** 舞伴经验对粉丝投票有轻微负向影响！可能原因：
- 老牌舞者可能与"过气"名人配对
- 粉丝可能更支持新鲜组合

### 3.3 Top Professional Dancers Ranking

基于平均选手名次排名（至少3次出场）：

| Rank | Partner | Appearances | Avg Placement | Best | Avg Judge | Win Rate |
|------|---------|-------------|---------------|------|-----------|----------|
| 🥇 | **Derek Hough** | 17 | 2.94 | 1 | 28.7 | 35% |
| 🥈 | Julianne Hough | 5 | 4.20 | 1 | 23.6 | 40% |
| 🥉 | Daniella Karagach | 5 | 4.60 | 1 | 27.6 | 40% |
| 4 | Mark Ballas | 21 | 5.19 | 1 | 27.1 | 10% |
| 5 | Valentin Chmerkovskiy | 19 | 5.26 | 1 | 28.7 | 11% |
| 6 | Lindsay Arnold | 10 | 5.40 | 1 | 25.6 | 10% |
| 7 | Witney Carson | 14 | 5.50 | 1 | 28.1 | 7% |
| 8 | Cheryl Burke | 25 | 5.80 | 1 | 24.3 | 8% |
| 9 | Maksim Chmerkovskiy | 17 | 6.00 | 1 | 25.1 | 6% |
| 10 | Sasha Farber | 12 | 6.08 | 3 | 28.0 | 0% |

**Derek Hough** 以平均名次2.94领先所有舞者，是DWTS历史上最成功的职业舞者。

---

## Part 4: Effect Comparison - Judges vs Fans

### 4.1 Standardized Coefficient Comparison

| Variable | Judge t-stat | Fan t-stat | Who Values More? |
|----------|--------------|------------|------------------|
| Age | **-3.84** | -2.20 | **Judges** (技术要求) |
| Age² | 2.01 | 0.68 | Judges |
| Is US | 0.16 | -0.28 | Neither (not significant) |
| **Celebrity Popularity** | -1.18 | **2.85** | **Fans** (明星效应) |
| Partner Popularity | 1.41 | **-2.88** | **Fans** (反向效应) |

### 4.2 Key Comparison: Partner Effect (ICC)

| Model | ICC | Interpretation |
|-------|-----|----------------|
| Judge Score | 13.66% | 舞者选择中度影响评委分 |
| Fan Vote | 15.26% | 舞者选择略微更影响粉丝投票 |

**结论：** 舞者选择对两者影响相似，但略微更影响粉丝投票。

### 4.3 Correlation Analysis

$$\rho_{Judge, Fan} = 0.5248 \quad (p < 0.001, n = 421)$$

**解读：** 评委分和粉丝投票呈中度正相关（r = 0.52），表明：
- 两者存在一定共识（舞技好的选手通常也受粉丝欢迎）
- 但也存在系统性差异（约73%的方差独立）

---

## Part 5: Key Findings Summary

### ✅ Finding 1: Age Effect is Significant
- 年龄对评委打分有显著负向影响
- 每增加10岁，评委分下降约3.4分
- 年龄对粉丝投票影响较小

### ✅ Finding 2: Popularity Affects Fans, Not Judges
- 名人社交媒体人气显著提升粉丝投票 (p < 0.01)
- 但对评委打分无显著影响 (p = 0.24)
- **结论：评委更"客观"，粉丝受名人光环影响**

### ✅ Finding 3: No Home Country Advantage
- 美国出生选手与国际选手无显著差异
- 评委和粉丝均未表现出国籍偏好

### ✅ Finding 4: Professional Dancer Matters
- 职业舞者选择解释13.7%-15.3%的结果方差
- 这是一个**中等程度**的效应
- Derek Hough是历史上最成功的舞伴

### ✅ Finding 5: Partner Experience is Double-Edged
- 舞伴经验正向影响评委分 (+0.2分/季)
- 但轻微负向影响粉丝投票
- 可能反映粉丝对"新鲜感"的偏好

---

## Part 6: Figures Generated

以下图表已保存到 `figures/` 目录：

| Figure | Description |
|--------|-------------|
| `task3_age_effect.png` | 年龄效应散点图（评委分 vs 粉丝投票） |
| `task3_popularity_effect.png` | 人气效应散点图 |
| `task3_partner_ranking.png` | 职业舞者排名条形图 |
| `task3_judge_fan_scatter.png` | 评委分 vs 粉丝投票散点图 |
| `task3_coefficient_comparison.png` | 系数比较柱状图 |
| `task3_icc_comparison.png` | ICC比较图 |
| `task3_partner_experience.png` | 舞伴经验效应箱线图 |

---

## Part 7: Limitations and Future Work

### Limitations:
1. **人气数据缺失**：部分选手社交媒体人气为0（数据未采集或不存在）
2. **行业分类缺失**：未包含选手职业类别（Actor, Athlete等）
3. **临时舞伴未处理**：某些周的临时替换舞伴未单独建模
4. **时间趋势未控制**：未考虑不同赛季的评分标准变化

### Future Work:
1. 添加选手职业类别作为控制变量
2. 使用面板数据模型（周层面）进行更细粒度分析
3. 引入交互效应（如：人气 × 年龄）
4. 使用机器学习方法进行特征重要性排序

---

## Appendix: Model Diagnostics

### OLS Assumptions Check:
- ✅ Linearity: Scatter plots show reasonable linear trends
- ✅ Independence: Cross-sectional data, observations independent
- ⚠️ Homoscedasticity: Should check residual plots
- ✅ Normality: Sample size large enough for CLT

### Mixed Model:
- ✅ Convergence: Both models converged successfully
- ✅ Random effects: Variance components positive and meaningful
- ⚠️ Should check random effect distribution

---

*End of Report*
