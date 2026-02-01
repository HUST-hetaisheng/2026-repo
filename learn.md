# 2026 MCM C 题 · 第三问 (Task 3)
# 建模思路与操作步骤 - 完整版

> 更新日期: 2026-02-01
> 新增内容: BMI、舞蹈经验、社交媒体综合指标、Google搜索热度

---

## 一、问题回顾与核心目标

### 1.1 原题要求

> **Task 3: Impact of Contestant and Partner Characteristics**
> 
> Use the data including your fan vote estimates to develop a model that 
> analyzes the impact of various pro dancers as well as characteristics for 
> the celebrities available in the data (age, industry, etc). How much do 
> such things impact how well a celebrity will do in the competition? 
> Do they impact judges scores and fan votes in the same way?

### 1.2 核心问题拆解

| 子问题 | 具体内容 | 关键挑战 |
|--------|----------|----------|
| **Q1** | 明星特征如何影响表现？ | 多维特征、非线性效应 |
| **Q2** | 职业舞者如何影响表现？ | 重复出现、临时替换、历史累积效应 |
| **Q3** | 对评委分和粉丝投票影响相同吗？ | 需要对比分析 |
| **Q4** | 国家/地区是否有影响？ | 美国本土 vs 国际选手 |
| **Q5** | 社交媒体人气如何影响结果？ | 人气→粉丝投票的传导机制 |

---

## 二、数据资源

### 2.1 原始数据字段

**明星特征 (Celebrity):**
| 字段名 | 类型 | 说明 | 数据来源 |
|--------|------|------|----------|
| `celebrity_industry` | 类别 | 职业 (Actor, Athlete, Singer, etc.) | 原始数据 |
| `celebrity_age_during_season` | 数值 | 参赛年龄 | 原始数据 |
| `celebrity_homestate` | 类别 | 家乡州 (美国选手) | 原始数据 |
| `celebrity_homecountry/region` | 类别 | 国家/地区 | 原始数据 |
| `Celebrity_Average_Popularity_Score` | 数值 | **社交媒体人气指数** | **爬取数据** |

**职业舞者 (Partner):**
| 字段名 | 类型 | 说明 | 数据来源 |
|--------|------|------|----------|
| `ballroom_partner` | 类别 | 舞伴姓名 | 原始数据 |
| `ballroom_partner_Average_Popularity_Score` | 数值 | **舞者人气指数** | **爬取数据** |

### 2.2 新增爬取数据字段 (2026-02-01)

**选手身体素质与经验 (来自 `crawl_celebrity_detailed_info.csv`):**
| 字段名 | 类型 | 说明 | 统计摘要 |
|--------|------|------|----------|
| `bmi` | 数值 | 身体质量指数 (Body Mass Index) | Mean=22.3, Range=[18, 28.4] |
| `dance_experience_score` | 数值 | 舞蹈经验评分 (0-10) | Mean=3.2, Range=[0, 10] |

**社交媒体周级数据 (来自 `crawl_social_media_weekly_data.csv`):**
| 字段名 | 类型 | 说明 | 处理方式 |
|--------|------|------|----------|
| `social_media_popularity` | 数值 | 社交媒体综合指数 | 由 Twitter Mentions + Instagram Engagement + Facebook Shares + YouTube Views **标准化后求和**生成 |
| `google_search_volume` | 数值 | Google 搜索热度 | 原始值 |

> **注意**: `social_media_popularity` 是**周级别变量**（每周变化），而 `bmi` 和 `dance_experience_score` 是**选手级别常量**（整个赛季不变）。

### 2.3 需要构造的派生变量

**明星层面:**
```python
# 国家/地区分组
is_us_born = 1 if country == 'United States' else 0
region_group = categorize_region(country)  # US, Europe, Other

# 人气变量（经测试原始值效果更优）
popularity = popularity  # 使用原始 Popularity Score，不进行 Log 变换
```

**舞者层面 (需特殊处理):**
```python
# 历史经验统计 - 截至当前赛季之前
partner_seasons_count = count(seasons before current)
partner_avg_placement = mean(placements before current)
partner_best_placement = min(placements before current)
partner_win_count = count(1st place before current)

# 处理临时替换
# 方案: 使用"主舞伴"（出场次数最多的舞伴）进行分析
```

### 2.4 因变量

| 变量 | 计算方法 | 说明 |
|------|----------|------|
| `placement` | 原始数据 | 最终名次 (1=冠军) |
| `avg_judge_score` | mean(周评委总分) | 平均评委分 |
| `avg_fan_vote_share` | mean(周粉丝份额) | 平均粉丝投票份额 |
| `weeks_survived` | max(week) where score > 0 | 存活周数 |

### 2.5 因变量选择决策 (2026-02-01)

**问题**: 综合因变量 $Y_3$ 应该用 `placement`（最终名次）还是构造 `composite_score = j + f`（评委份额+粉丝份额相加）？

**分析**:

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Placement** | 简洁、完整、所有季节可比 | 只有排名，无量级信息 |
| **Composite Score** | S3-27 符合官方规则，信息量更大 | S1-2 (Rank) 和 S28-34 (Bottom2) 规则不同，需要映射 |

**决策**: 选择 **Placement** 作为 $Y_3$，原因如下：

1. **简洁性**: 三个因变量各有明确含义，避免模型过于复杂
   - $Y_1$ = `avg_judge_score`：专业评价（评委怎么看）
   - $Y_2$ = `avg_fan_vote_share`：观众偏好（粉丝怎么投）
   - $Y_3$ = `placement`：综合结果（谁赢了）

2. **论文写作友好**: 分别建模更易于叙述和解释

3. **避免额外假设**: 使用 Composite Score 需要对 S1-2 和 S28-34 做 Rank→Score 映射，引入 Gibbs Softmax 等额外假设

4. **对比分析完整**: 通过 $Y_1$ 和 $Y_2$ 的系数对比，已经能回答"评委和粉丝看重的因素是否相同"这一核心问题

---

## 三、分析框架（两大模块）

### 3.1 模块A：明星特征分析

**目标**: 量化明星自身特征对表现的影响

#### Industry 变量处理

原始数据有 26 个行业类别，但分布极不均匀：
- **高频类别（保留）**: Actor/Actress (128), Athlete (95), TV Personality (67), Singer/Rapper (61), Model (17), Comedian (12), Social Media Personality (9)
- **低频类别（合并为 Other）**: Entrepreneur, Racing Driver, Politician, News Anchor, 等共 32 人

**合并后**：8 个类别，7 个虚拟变量，样本量充足（421人 >> 7变量），**不存在维度灾难**。

```python
# Industry 类别合并
industry_mapping = {
    'Actor/Actress': 'Actor',
    'Athlete': 'Athlete', 
    'TV Personality': 'TV',
    'Singer/Rapper': 'Singer',
    'Model': 'Model',
    'Comedian': 'Comedian',
    'Social Media Personality': 'SocialMedia',
    'Social media personality': 'SocialMedia',  # 合并大小写
}
df['industry_group'] = df['celebrity_industry'].map(
    lambda x: industry_mapping.get(x, 'Other')
)
```

#### 模型公式

**基础模型 (Baseline):**

$$
Y_i = \beta_0 + \underbrace{\beta_1 \cdot \text{Age}_i + \beta_2 \cdot \text{Age}_i^2}_{\text{年龄效应（含非线性）}} 
    + \underbrace{\delta \cdot \mathbf{1}[\text{US}_i]}_{\text{国家效应}} 
    + \underbrace{\phi \cdot \text{Popularity}_i}_{\text{人气效应}} 
    + \epsilon_i
$$

**扩展模型 (含行业效应与新增变量):**

$$
Y_i = \beta_0 + \beta_1 \cdot \text{Age}_i + \beta_2 \cdot \text{Age}_i^2 
    + \beta_3 \cdot \mathbf{1}[\text{US}_i] 
    + \beta_4 \cdot \text{Popularity}_i 
    + \beta_5 \cdot \text{BMI}_i
    + \beta_6 \cdot \text{DanceExp}_i 
    + \underbrace{\sum_{k=1}^{7} \gamma_k \cdot \mathbf{1}[\text{Industry}_i = k]}_{\text{行业虚拟变量（基准类=Other）}} 
    + \epsilon_i
$$

其中行业虚拟变量 $k \in \{\text{Actor, Athlete, TV, Singer, Model, Comedian, SocialMedia}\}$，基准类为 Other。

**周级别扩展模型 (含社交媒体动态效应):**

$$
Y_{it} = \beta_0 + \beta_1 \cdot \text{Age}_i + \beta_2 \cdot \mathbf{1}[\text{US}_i] 
    + \beta_3 \cdot \text{BMI}_i + \beta_4 \cdot \text{DanceExp}_i 
    + \sum_{k} \gamma_k \cdot \mathbf{1}[\text{Industry}_i = k]
    + \underbrace{\lambda_1 \cdot \text{SocialPop}_{it}}_{\text{周社交媒体热度}} 
    + \underbrace{\lambda_2 \cdot \text{GoogleSearch}_{it}}_{\text{周搜索热度}} 
    + \epsilon_{it}
$$

其中：
- $i$ 索引选手，$t$ 索引比赛周次
- $\text{SocialPop}_{it}$ = 第 $i$ 位选手在第 $t$ 周的社交媒体综合指数（标准化后求和）
- $\text{GoogleSearch}_{it}$ = 第 $i$ 位选手在第 $t$ 周的 Google 搜索热度

> **选手级 vs 周级别模型的区别**：
> - **选手级模型**：样本单位是选手（N≈421），回答"什么样的选手更成功？"
> - **周级别模型**：样本单位是选手-周（N≈2777），回答"什么因素影响每周表现？"
> - 静态变量（Age, Industry 等）在两个模型中的系数含义相同，但周级别模型的标准误更小
> - 动态变量（SocialPop, GoogleSearch）只在周级别模型中有意义

**因变量 $Y$ 可以是：**
- 评委打分模型: $Y = \bar{J}_i$ (平均评委分)
- 粉丝投票模型: $Y = \bar{F}_i$ (平均粉丝投票份额)
- 最终名次模型: $Y = \text{Placement}_i$

### 3.2 模块B：职业舞者分析（独立模块）

**为什么需要独立分析？**

1. **重复出现**: 同一舞者在多个赛季出现多次，样本不独立
2. **历史累积**: 舞者的能力通过历史表现体现
3. **临时替换**: 某些周会有临时舞伴，需要清洗
4. **样本量不均**: 有些舞者仅出现1-2次，有些出现20+次

**分析方法:**

| 方法 | 适用场景 | 优点 |
|------|----------|------|
| **混合效应模型** | 舞者作为随机效应 | 处理重复测量、估计方差分量 |
| **固定效应回归** | 舞者作为虚拟变量 | 直接估计各舞者的边际贡献 |
| **舞者排名表** | 描述性分析 | 直观展示"最佳舞者"排名 |

**为什么用线性回归 (OLS) 而不是 Logit？**

| 因变量 | 类型 | 推荐模型 |
|--------|------|----------|
| `placement` (1,2,3,...) | 连续/有序数值 | **OLS** 或 有序Logit |
| `avg_judge_score` | 连续数值 (0-40) | **OLS** |
| `avg_fan_vote_share` | 连续比例 (0-1) | **OLS** 或 Beta回归 |
| `是否夺冠` (0/1) | 二元分类 | Logit |

**Logit 仅适用于二元分类变量**（如"是否夺冠"）。我们的因变量 `placement`、`avg_judge_score` 都是**连续数值**，所以 OLS 是合理选择。

**混合效应模型 (基础版):**

$$
Y_{ij} = \beta_0 + X_{ij}\beta + \underbrace{u_j}_{\text{舞者随机效应}} + \epsilon_{ij}
$$

**混合效应模型 (扩展版，含行业效应与新增变量):**

$$
Y_{ij} = \beta_0 + \beta_1 \cdot \text{Age}_{ij} + \beta_2 \cdot \mathbf{1}[\text{US}_{ij}] + \beta_3 \cdot \text{Popularity}_{ij} + \beta_4 \cdot \text{BMI}_{ij} + \beta_5 \cdot \text{DanceExp}_{ij} + \sum_{k} \gamma_k \cdot \mathbf{1}[\text{Industry}_{ij}=k] + \beta_6 \cdot \text{PartnerExp}_j + u_j + \epsilon_{ij}
$$

其中：
- $j$ 索引舞者，$i$ 索引该舞者的各次出场（即不同赛季带的不同选手）
- $Y_{ij}$ 是因变量（如 `placement` 或 `avg_judge_score`）
- $X_{ij}$ 是**选手级控制变量向量**，包含：
  - `Age`: 参赛年龄
  - `isUS`: 是否美国选手
  - `Industry`: 行业类别（7 个虚拟变量，基准类=Other）
  - `Popularity`: 人气指数（原始值）
  - `BMI`: 身体质量指数 (**新增**)
  - `DanceExp`: 舞蹈经验评分 (**新增**)
  - `PartnerExp`: 舞伴历史参赛次数
- $u_j \sim N(0, \sigma^2_u)$：舞者 $j$ 的随机效应，反映该舞者的"附加价值"
- $\epsilon_{ij} \sim N(0, \sigma^2_\epsilon)$：残差项

**周级别混合效应模型 (含社交媒体动态效应):**

$$
Y_{ijt} = \beta_0 + X_{ij}\beta + \underbrace{\gamma_1 \cdot \text{SocialPop}_{ijt} + \gamma_2 \cdot \text{GoogleSearch}_{ijt}}_{\text{周级别社交媒体效应}} + u_j + \epsilon_{ijt}
$$

其中：
- $t$ 索引比赛周次
- $\text{SocialPop}_{ijt}$：选手 $i$ 在第 $t$ 周的社交媒体综合指数（Twitter + Instagram + Facebook + YouTube 标准化求和）
- $\text{GoogleSearch}_{ijt}$：选手 $i$ 在第 $t$ 周的 Google 搜索热度

**$X_{ij}$ 的具体例子（含新增变量）：**

假设数据如下：

| 舞者 $j$ | 赛季 | 选手 | Age | Popularity | isUS | BMI | DanceExp | PartnerExp | Placement |
|----------|------|------|-----|------------|------|-----|----------|------------|-----------|
| Derek Hough | S10 | Nicole Scherzinger | 32 | 50 | 1 | 21.5 | 7.2 | 4 | 1 |
| Derek Hough | S11 | Jennifer Grey | 50 | 30 | 1 | 20.8 | 8.5 | 5 | 1 |
| Derek Hough | S14 | Maria Menounos | 33 | 20 | 1 | 22.1 | 2.0 | 6 | 4 |
| Cheryl Burke | S2 | Drew Lachey | 29 | 15 | 1 | 23.0 | 1.5 | 0 | 1 |
| Cheryl Burke | S3 | Emmitt Smith | 37 | 25 | 1 | 24.5 | 9.0 | 1 | 1 |

**对于 Derek Hough (j=Derek) 在 S10 带 Nicole (i=1)：**

$$
X_{1,\text{Derek}} = [\text{Age}=32,\ \text{Industry}=\text{Singer},\ \text{Popularity}=50,\ \text{isUS}=1,\ \text{BMI}=21.5,\ \text{DanceExp}=7.2,\ \text{PartnerExp}=4]
$$

**对于 Derek Hough 在 S11 带 Jennifer Grey (i=2)：**

$$
X_{2,\text{Derek}} = [\text{Age}=50,\ \text{Industry}=\text{Actor},\ \text{Popularity}=30,\ \text{isUS}=1,\ \text{BMI}=20.8,\ \text{DanceExp}=8.5,\ \text{PartnerExp}=5]
$$

**模型完整展开（以 Placement 为因变量，含行业效应与新增变量）：**

$$
\text{Placement}_{ij} = \beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{Popularity} + \beta_3 \cdot \text{isUS} + \beta_4 \cdot \text{BMI} + \beta_5 \cdot \text{DanceExp} + \sum_k \gamma_k \cdot \mathbf{1}[\text{Industry}=k] + \beta_6 \cdot \text{PartnerExp} + u_j + \epsilon_{ij}
$$

**系数解读：**
| 系数 | 变量 | 预期方向 | 解释 |
|------|------|----------|------|
| $\beta_1$ | Age | 不确定 | 年龄对名次的影响（可能有二次项） |
| $\beta_2$ | Popularity | 负 | 人气越高，名次越靠前（数值越小） |
| $\beta_3$ | isUS | 负 | 美国选手可能有主场优势 |
| $\beta_4$ | BMI | 不确定 | 身体素质对舞蹈表现的影响 |
| $\beta_5$ | DanceExp | **负** | **舞蹈经验越丰富，名次越靠前** |
| $\gamma_{\text{Athlete}}$ | Industry=Athlete | **负** | 运动员身体素质好，预期名次靠前 |
| $\gamma_{\text{Singer}}$ | Industry=Singer | 负 | 歌手节奏感强，有舞台经验 |
| $\gamma_{\text{TV}}$ | Industry=TV | 不确定 | 知名度高但舞技一般 |
| $\beta_6$ | PartnerExp | 负 | 舞伴经验越丰富，名次越靠前 |
| $u_j$ | 舞者随机效应 | - | 舞者 $j$ 的"教学附加价值" |

**模型逻辑：** 控制选手特征 $X_{ij}$（包括行业类别）后，$u_j$ 反映的是"同样特征的选手，配对不同舞者后名次的差异"——即舞者的真实边际贡献。如果 $u_{\text{Derek}} < 0$（名次越小越好），说明 Derek 能让选手名次更好。

**各变量的预期作用：**
- **Industry**: 不同行业的选手有不同的先天优势（运动员体能好、歌手节奏感强、演员表现力佳）
- **BMI**: 舞蹈是高强度运动，适中的 BMI 可能有利于表现
- **DanceExp**: 有舞蹈背景的选手起点更高，预期显著负相关
- **SocialPop** (周级别): 社交媒体热度可能直接影响粉丝投票
- **GoogleSearch** (周级别): 搜索热度反映公众关注度，可能与粉丝投票正相关

---

## 四、具体操作步骤

### Step 1: 数据准备

```python
import pandas as pd
import numpy as np

# 1.1 加载数据
raw_df = pd.read_csv('2026_MCM_Problem_C_Data_Cleaned.csv')
pop_df = pd.read_csv('2026_MCM_Problem_C_Data_Cleaned添加人气后.csv')
fan_df = pd.read_csv('fan_vote_results_final.csv')

# 1.2 合并人气数据
# pop_df 包含: Celebrity_Average_Popularity_Score, ballroom_partner_Average_Popularity_Score

# 1.3 处理国家/地区
def categorize_region(country):
    if pd.isna(country):
        return 'Unknown'
    elif country == 'United States':
        return 'US'
    elif country in ['England', 'United Kingdom', 'Ireland', 'Scotland', 'Australia', 'New Zealand', 'Canada']:
        return 'English-Speaking'
    else:
        return 'Other'

raw_df['region_group'] = raw_df['celebrity_homecountry/region'].apply(categorize_region)
raw_df['is_us'] = (raw_df['celebrity_homecountry/region'] == 'United States').astype(int)

# 1.4 人气数据处理
pop_df['celeb_pop'] = pop_df['Celebrity_Average_Popularity_Score']
pop_df['partner_pop'] = pop_df['ballroom_partner_Average_Popularity_Score']
```

### Step 2: 选手级汇总统计

```python
# 2.1 计算每位选手的汇总统计
def compute_contestant_summary(raw_df, fan_df):
    # 评委分统计
    judge_cols = [c for c in raw_df.columns if 'judge' in c and 'score' in c]
    
    summary = []
    for idx, row in raw_df.iterrows():
        # 计算每周总分
        weekly_totals = []
        for w in range(1, 12):
            week_cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            week_cols = [c for c in week_cols if c in raw_df.columns]
            scores = [row[c] for c in week_cols if pd.notna(row[c]) and row[c] > 0]
            if scores:
                weekly_totals.append(sum(scores))
        
        avg_judge = np.mean(weekly_totals) if weekly_totals else 0
        weeks_survived = len(weekly_totals)
        
        summary.append({
            'celebrity_name': row['celebrity_name'],
            'season': row['season'],
            'avg_judge_score': avg_judge,
            'weeks_survived': weeks_survived,
            'placement': row['placement']
        })
    
    summary_df = pd.DataFrame(summary)
    
    # 合并粉丝投票统计
    fan_summary = fan_df.groupby(['season', 'celebrity_name']).agg({
        'fan_vote_share': 'mean',
        'cv': 'mean'
    }).reset_index()
    fan_summary.columns = ['season', 'celebrity_name', 'avg_fan_vote_share', 'avg_cv']
    
    return summary_df.merge(fan_summary, on=['season', 'celebrity_name'], how='left')

contestant_df = compute_contestant_summary(raw_df, fan_df)
```

### Step 3: 舞者历史统计（关键！）

```python
# 3.1 构建舞者历史表现表
def compute_partner_history(raw_df):
    """
    对于每个 (season, partner) 组合，计算该舞者截至本赛季之前的历史统计
    """
    # 按舞者和赛季汇总
    partner_season = raw_df.groupby(['ballroom_partner', 'season']).agg({
        'placement': 'min',  # 一个赛季只带一个选手
        'celebrity_name': 'first'
    }).reset_index()
    
    history_records = []
    
    for partner in partner_season['ballroom_partner'].unique():
        partner_data = partner_season[partner_season['ballroom_partner'] == partner].sort_values('season')
        
        # 累积统计
        placements_before = []
        for idx, row in partner_data.iterrows():
            current_season = row['season']
            
            history_records.append({
                'ballroom_partner': partner,
                'season': current_season,
                'partner_seasons_before': len(placements_before),
                'partner_avg_placement_before': np.mean(placements_before) if placements_before else np.nan,
                'partner_best_placement_before': min(placements_before) if placements_before else np.nan,
                'partner_wins_before': sum(1 for p in placements_before if p == 1)
            })
            
            placements_before.append(row['placement'])
    
    return pd.DataFrame(history_records)

partner_history = compute_partner_history(raw_df)
```

### Step 4: 处理临时替换

```python
# 4.1 识别临时替换情况
# 方法: 某周的舞伴与其他周不同 → 使用"主舞伴"

def identify_primary_partner(raw_df):
    """
    对于存在临时替换的情况，识别主舞伴
    目前数据中 ballroom_partner 是赛季级别的，暂不需要特殊处理
    如果有周级别的舞伴数据，需要用众数确定主舞伴
    """
    # 当前数据结构: 每行是一个选手-赛季，ballroom_partner 是整季的
    # 如果某周有临时替换，原始数据可能未体现
    # 保守处理: 使用原始的 ballroom_partner
    return raw_df[['celebrity_name', 'season', 'ballroom_partner']]

# 注: 如果数据中有周级别舞伴变化，需要额外清洗逻辑
```

### Step 5: 模块A - 明星特征回归分析

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm

# 5.1 合并所有特征
analysis_df = contestant_df.merge(
    raw_df[['celebrity_name', 'season', 'celebrity_industry', 
            'celebrity_age_during_season', 'celebrity_homecountry/region', 'is_us', 'region_group']],
    on=['celebrity_name', 'season']
).merge(
    pop_df[['celebrity_name', 'season', 'celeb_pop']],
    on=['celebrity_name', 'season']
)

# 5.2 评委打分模型
model_judge = ols("""
    avg_judge_score ~ celebrity_age_during_season 
                    + I(celebrity_age_during_season**2)
                    + is_us 
                    + celeb_pop
""", data=analysis_df).fit()

print("=== 评委打分影响因素 ===")
print(model_judge.summary())

# 5.3 粉丝投票模型
model_fan = ols("""
    avg_fan_vote_share ~ celebrity_age_during_season 
                       + I(celebrity_age_during_season**2)
                       + is_us 
                       + celeb_pop
""", data=analysis_df).fit()

print("=== 粉丝投票影响因素 ===")
print(model_fan.summary())

# 5.4 最终名次模型 (名次越小越好，需要注意解读)
model_placement = ols("""
    placement ~ celebrity_age_during_season 
              + I(celebrity_age_during_season**2)
              + is_us 
              + celeb_pop
""", data=analysis_df).fit()
```

### Step 6: 模块B - 职业舞者分析（独立）

```python
# 6.1 合并舞者历史数据
partner_analysis_df = analysis_df.merge(
    partner_history,
    on=['ballroom_partner', 'season']
).merge(
    pop_df[['celebrity_name', 'season', 'partner_pop']],
    on=['celebrity_name', 'season']
)

# 6.2 方法1: 舞者固定效应模型
# 估计每个舞者的"附加价值"
model_partner_fe = ols("""
    placement ~ celebrity_age_during_season 
              + celeb_pop
              + C(ballroom_partner)
""", data=partner_analysis_df).fit()

# 提取舞者系数
partner_effects = model_partner_fe.params.filter(like='ballroom_partner')
partner_effects = partner_effects.sort_values()

# 6.3 方法2: 混合效应模型
# 舞者作为随机效应，估计舞者效应的方差
model_partner_re = mixedlm("""
    placement ~ celebrity_age_during_season 
              + celeb_pop
              + partner_seasons_before
""", data=partner_analysis_df, groups='ballroom_partner').fit()

print("=== 舞者随机效应模型 ===")
print(model_partner_re.summary())

# 6.4 舞者排名表（描述性分析）
partner_ranking = partner_analysis_df.groupby('ballroom_partner').agg({
    'placement': ['mean', 'min', 'count'],
    'avg_judge_score': 'mean'
}).reset_index()
partner_ranking.columns = ['partner', 'avg_placement', 'best_placement', 'n_seasons', 'avg_judge_score']
partner_ranking = partner_ranking[partner_ranking['n_seasons'] >= 3].sort_values('avg_placement')
```

### Step 7: 效应对比分析

```python
# 7.1 提取标准化系数进行对比
from scipy.stats import zscore

# 标准化连续变量
for col in ['celebrity_age_during_season', 'celeb_pop']:
    analysis_df[f'{col}_z'] = zscore(analysis_df[col].fillna(0))

# 重新拟合标准化模型
model_judge_std = ols("""
    avg_judge_score ~ celebrity_age_during_season_z 
                    + is_us 
                    + celeb_pop_z
""", data=analysis_df).fit()

model_fan_std = ols("""
    avg_fan_vote_share ~ celebrity_age_during_season_z 
                       + is_us 
                       + celeb_pop_z
""", data=analysis_df).fit()

# 7.2 比较同一变量在两个模型中的系数
comparison = pd.DataFrame({
    'Variable': ['Age (std)', 'US Born', 'Popularity (std)'],
    'Judge Score Effect': [
        model_judge_std.params.get('celebrity_age_during_season_z', 0),
        model_judge_std.params.get('is_us', 0),
        model_judge_std.params.get('celeb_pop_z', 0)
    ],
    'Fan Vote Effect': [
        model_fan_std.params.get('celebrity_age_during_season_z', 0),
        model_fan_std.params.get('is_us', 0),
        model_fan_std.params.get('celeb_pop_z', 0)
    ]
})
comparison['Difference'] = comparison['Fan Vote Effect'] - comparison['Judge Score Effect']
comparison['Ratio'] = comparison['Fan Vote Effect'] / comparison['Judge Score Effect']
```

### Step 8: 可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 8.1 职业类别 vs 评委分/粉丝投票
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 评委分
industry_judge = analysis_df.groupby('celebrity_industry')['avg_judge_score'].mean().sort_values()
axes[0].barh(industry_judge.index, industry_judge.values)
axes[0].set_title('Average Judge Score by Industry')

# 粉丝投票
industry_fan = analysis_df.groupby('celebrity_industry')['avg_fan_vote_share'].mean().sort_values()
axes[1].barh(industry_fan.index, industry_fan.values)
axes[1].set_title('Average Fan Vote Share by Industry')

plt.tight_layout()
plt.savefig('figures/task3_industry_comparison.png', dpi=300)

# 8.2 年龄效应曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 年龄 vs 评委分
sns.regplot(x='celebrity_age_during_season', y='avg_judge_score', 
            data=analysis_df, ax=axes[0], order=2, scatter_kws={'alpha':0.5})
axes[0].set_title('Age vs Judge Score')

# 年龄 vs 粉丝投票
sns.regplot(x='celebrity_age_during_season', y='avg_fan_vote_share', 
            data=analysis_df, ax=axes[1], order=2, scatter_kws={'alpha':0.5})
axes[1].set_title('Age vs Fan Vote Share')

plt.savefig('figures/task3_age_effect.png', dpi=300)

# 8.3 人气 vs 粉丝投票（关键图！）
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='celeb_pop', y='avg_fan_vote_share', 
                hue='celebrity_industry', data=analysis_df, ax=ax)
ax.set_xlabel('Popularity Score')
ax.set_ylabel('Average Fan Vote Share')
ax.set_title('Social Media Popularity vs Fan Votes')
plt.savefig('figures/task3_popularity_fanvote.png', dpi=300)

# 8.4 舞者排名条形图
fig, ax = plt.subplots(figsize=(10, 8))
top_partners = partner_ranking.head(15)
ax.barh(top_partners['partner'], top_partners['avg_placement'])
ax.invert_xaxis()  # 排名越低越好
ax.set_xlabel('Average Placement (lower is better)')
ax.set_title('Top 15 Professional Dancers by Average Placement')
plt.savefig('figures/task3_partner_ranking.png', dpi=300)

# 8.5 效应对比森林图
fig, ax = plt.subplots(figsize=(10, 6))
variables = comparison['Variable']
y_pos = range(len(variables))

ax.barh([y - 0.2 for y in y_pos], comparison['Judge Score Effect'], 
        height=0.35, label='Judge Score', color='steelblue')
ax.barh([y + 0.2 for y in y_pos], comparison['Fan Vote Effect'], 
        height=0.35, label='Fan Vote', color='coral')

ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Standardized Effect')
ax.set_title('Comparison of Effects on Judge Score vs Fan Vote')
ax.legend()

plt.savefig('figures/task3_effect_comparison.png', dpi=300)
```

---

## 五、关键分析问题与检验方法

### 5.1 社交媒体人气是否影响粉丝投票？

**假设**: 人气高的选手获得更多粉丝投票

**检验方法:**
1. 相关性分析: `corr(popularity, avg_fan_vote_share)`
2. 回归系数检验: $H_0: \phi = 0$ (人气系数为0)
3. 分组比较: 高人气 vs 低人气选手的平均投票份额

**预期结果**: 人气对粉丝投票有显著正效应，对评委分无显著效应

### 5.2 国家/地区是否影响结果？

**假设**: 美国本土选手可能获得更多粉丝支持（本土优势）

**检验方法:**
1. 分组 t 检验: 美国 vs 非美国选手的平均投票份额
2. 回归中的 is_us 系数显著性检验
3. 按地区分组的箱线图可视化

**细分分析:**
- English-Speaking 国家 vs 其他国家
- 按州分析（人口大州是否有优势？）

### 5.3 职业舞者的贡献有多大？

**检验方法:**
1. **方差分解**: 舞者随机效应的方差 / 总方差
2. **模型比较**: 有/无舞者效应的模型 R² 差异
3. **舞者排名**: 控制选手特征后的舞者边际贡献

**ICC (类内相关系数):**
$$
\text{ICC} = \frac{\sigma^2_{\text{partner}}}{\sigma^2_{\text{partner}} + \sigma^2_{\epsilon}}
$$

如果 ICC 高，说明舞者效应解释了结果的很大一部分变异。

### 5.4 同一因素对评委分和粉丝投票影响是否相同？

**检验方法:**

**方法A: 系数比较**
- 标准化两个模型，比较同一变量的系数大小

**方法B: 交互效应模型（堆叠数据）**
```python
# 将评委分和粉丝投票堆叠成长格式
stacked_df = pd.melt(analysis_df, 
                     id_vars=['celebrity_name', 'celebrity_age_during_season', ...],
                     value_vars=['avg_judge_score', 'avg_fan_vote_share'],
                     var_name='outcome_type', value_name='outcome')

# 拟合交互模型
model_interaction = ols("""
    outcome ~ celebrity_age_during_season * C(outcome_type)
            + celeb_pop * C(outcome_type)
            + is_us * C(outcome_type)
""", data=stacked_df).fit()

# 交互项系数显著 → 该因素对两种结果的影响不同
```

---

## 六、预期发现（假设）

### 6.1 社交媒体人气

| 假设 | 预期效应 |
|------|----------|
| 人气 → 评委分 | **无显著效应** (评委理论上独立评分) |
| 人气 → 粉丝投票 | **显著正效应** (粉丝基础大 → 投票多) |
| 人气 → 最终名次 | **显著负效应** (人气高 → 排名好) |

### 6.2 职业类别

| 职业 | 评委分 | 粉丝投票 | 解释 |
|------|--------|----------|------|
| Athlete | 较高 | 中等 | 身体素质好，但粉丝群可能不广 |
| Actor/Actress | 中等 | 较高 | 表演力强，知名度高 |
| Singer/Rapper | 中等 | 较高 | 节奏感好，粉丝忠诚 |
| TV Personality | 较低 | **很高** | 舞技一般，但粉丝互动多 |
| Model | 中等 | 中等 | 视觉效果好 |

### 6.3 年龄效应

- 评委分: 倒U型 (30-45岁最佳，太年轻缺表现力，太老体力下降)
- 粉丝投票: 可能与知名度正相关（中年明星往往更出名）

### 6.4 国家/地区效应

- 美国本土选手: 粉丝投票优势 (同胞支持)
- 评委分: 无显著国家差异 (假设评委公正)

### 6.5 职业舞者效应

- 经验丰富的舞者: 更高的评委分 (教学能力强)
- 明星舞者 (如 Derek Hough, Cheryl Burke): 自带粉丝，拉高投票

---

## 七、输出文件规划

### 7.1 数据文件
```
data/
├── task3_analysis_df.csv           # 分析用数据集
├── task3_partner_history.csv       # 舞者历史统计
├── task3_regression_judge.csv      # 评委分回归结果
├── task3_regression_fan.csv        # 粉丝投票回归结果
├── task3_partner_ranking.csv       # 舞者排名
└── task3_effect_comparison.csv     # 效应对比表
```

### 7.2 图表文件
```
figures/
├── task3_industry_comparison.png   # 职业类别对比
├── task3_age_effect.png            # 年龄效应曲线
├── task3_popularity_fanvote.png    # 人气 vs 粉丝投票
├── task3_partner_ranking.png       # 舞者排名
├── task3_effect_comparison.png     # 效应对比森林图
├── task3_region_effect.png         # 国家/地区效应
└── task3_partner_variance.png      # 舞者效应方差分解
```

### 7.3 代码文件
```
code/
├── task3_data_preparation.py       # 数据准备与特征工程
├── task3_celebrity_analysis.py     # 模块A: 明星特征分析
├── task3_partner_analysis.py       # 模块B: 舞者独立分析
├── task3_effect_comparison.py      # 效应对比分析
└── task3_visualization.py          # 可视化
```

---

## 八、论文写作大纲

### Section: Impact of Contestant and Partner Characteristics

#### 8.1 Introduction & Research Questions (0.25 page)
- 核心问题: 什么因素决定谁能在DWTS中走得更远？
- 数据资源: 原始特征 + 爬取的社交媒体人气

#### 8.2 Celebrity Characteristics Analysis (1 page)
- 8.2.1 职业类别效应 (表 + 条形图)
- 8.2.2 年龄效应 (二次项显著性)
- 8.2.3 国家/地区效应 (美国本土优势？)
- 8.2.4 社交媒体人气效应 (关键发现!)

#### 8.3 Professional Dancer Analysis (0.75 page)
- 8.3.1 舞者历史表现与选手结果
- 8.3.2 舞者随机效应模型 (ICC解读)
- 8.3.3 "最佳舞者"排名表

#### 8.4 Differential Impact: Judges vs Fans (0.5 page)
- 核心对比: 哪些因素对评委分和粉丝投票影响不同？
- 人气: 只影响粉丝投票
- 职业类别: TV Personality 对粉丝的影响 > 对评委
- 年龄/技术因素: 对评委影响 > 对粉丝

#### 8.5 Key Findings Summary (0.25 page)
- 人气是粉丝投票的强预测因子
- 舞者效应解释了约 X% 的结果变异
- 美国本土选手有 Y% 的投票优势

---

## 九、执行时间估计

| 步骤 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 数据准备与合并 | 1.5 小时 |
| 2 | 舞者历史统计计算 | 1 小时 |
| 3 | 明星特征回归分析 | 2 小时 |
| 4 | 舞者独立分析 | 1.5 小时 |
| 5 | 效应对比分析 | 1 小时 |
| 6 | 可视化 | 2 小时 |
| 7 | 论文写作 | 3 小时 |
| **总计** | | **12 小时** |

---

## 十、注意事项与潜在问题

### 10.1 数据质量
- 人气数据可能有缺失 (值为0): 需要区分"0人气"和"数据缺失"
- 早期赛季的选手可能没有社交媒体数据

### 10.2 统计问题
- **多重共线性**: 人气与职业类别可能相关
- **样本量**: 某些职业类别样本量小
- **选择偏差**: 能上节目的选手本身就不是随机抽样

### 10.3 舞者分析特殊问题
- 舞者出场次数不均 (有些仅1-2次)
- 舞者与选手匹配可能非随机 (节目组刻意安排)
- 舞者效应可能与赛季效应混淆

### 10.4 因果推断警告
- 这是观察研究，只能说明相关性
- 例如: "高人气选手获得更多投票"不意味着"增加人气可以增加投票"

---

## 十一、与其他任务的衔接

| 任务 | 与Task 3的关系 |
|------|----------------|
| Task 1 | 提供粉丝投票估计值作为因变量 |
| Task 2 | 规则分析 + 特征分析互补 (规则是游戏规则，特征是玩家能力) |
| Task 4 | 如果发现某些特征过度影响结果，可在新规则中"平衡" |

---

## 十二、总结

**Task 3 的核心逻辑:**

```
明星特征 (年龄、职业、国籍、人气) ──┬──→ 评委打分 ───┐
                                   │                ├──→ 最终名次
职业舞者效应 (经验、历史、人气) ────┴──→ 粉丝投票 ───┘
```

**关键创新点:**
1. 引入社交媒体人气数据，量化"人气→投票"的传导
2. 分离明星特征分析与舞者效应分析
3. 正式检验"评委分 vs 粉丝投票"的差异化影响

**一句话概括:**
> Task 3 回答"谁更可能赢？"——通过量化选手特征和舞伴效应
> 对评委评分与粉丝投票的差异化影响，揭示 DWTS 成功的关键因素，
> 并特别揭示社交媒体人气对粉丝投票的强预测作用。

---

# END OF DOCUMENT
