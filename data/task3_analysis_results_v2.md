# Task 3 Analysis Results (Version 2)
# Impact of Contestant and Partner Characteristics

*Generated: 2026-02-01 12:10*

*Data: task3_dataset_full.csv (with BMI, DanceExp, SocialPop, GoogleSearch)*

---

# Module A: Celebrity Characteristics Analysis (Contestant-Level OLS)

## Model Specification

**扩展模型公式:**

$$Y_i = \beta_0 + \beta_1 \cdot Age_i + \beta_2 \cdot Age_i^2 + \beta_3 \cdot isUS_i + \beta_4 \cdot Popularity_i + \beta_5 \cdot BMI_i + \beta_6 \cdot DanceExp_i + \sum_k \gamma_k \cdot Industry_k + \epsilon_i$$


**Sample Size:** 421 contestants with complete data


## Model 1: Average Judge Score

**R-squared:** 0.2370
**Adjusted R-squared:** 0.2107
**F-statistic:** 9.01 (p = 2.9597e-17)
**AIC:** 2533.1

| Variable | Coefficient | Std Error | t-value | p-value | Sig |
|----------|-------------|-----------|---------|---------|-----|
| Intercept | 31.5554 | 3.4978 | 9.02 | 0.0000 | *** |
| Industry_[T.Actor] | 1.8420 | 1.0098 | 1.82 | 0.0689 | * |
| Industry_[T.Athlete] | 0.5183 | 1.0249 | 0.51 | 0.6133 |  |
| Industry_[T.Comedian] | -0.3488 | 1.6445 | -0.21 | 0.8321 |  |
| Industry_[T.Model] | -2.4445 | 1.4871 | -1.64 | 0.1010 |  |
| Industry_[T.Singer] | 1.3587 | 1.2136 | 1.12 | 0.2636 |  |
| Industry_[T.SocialMedia] | 5.2972 | 1.8797 | 2.82 | 0.0051 | *** |
| Industry_[T.TV] | -0.3729 | 1.0563 | -0.35 | 0.7243 |  |
| age | -0.2770 | 0.0905 | -3.06 | 0.0024 | *** |
| age_squared | 0.0013 | 0.0010 | 1.26 | 0.2098 |  |
| is_us | -0.3117 | 0.7117 | -0.44 | 0.6616 |  |
| celeb_popularity | -0.0547 | 0.0740 | -0.74 | 0.4601 |  |
| partner_popularity | 0.0838 | 0.0822 | 1.02 | 0.3087 |  |
| bmi | 0.0471 | 0.1253 | 0.38 | 0.7072 |  |
| dance_experience_score | -0.1649 | 0.1409 | -1.17 | 0.2427 |  |

## Model 2: Average Fan Vote Share

**R-squared:** 0.1766
**Adjusted R-squared:** 0.1482
**F-statistic:** 6.22 (p = 2.7913e-11)
**AIC:** -1397.0

| Variable | Coefficient | Std Error | t-value | p-value | Sig |
|----------|-------------|-----------|---------|---------|-----|
| Intercept | 0.1757 | 0.0329 | 5.35 | 0.0000 | *** |
| Industry_[T.Actor] | 0.0057 | 0.0095 | 0.60 | 0.5467 |  |
| Industry_[T.Athlete] | 0.0038 | 0.0096 | 0.40 | 0.6918 |  |
| Industry_[T.Comedian] | -0.0150 | 0.0155 | -0.97 | 0.3336 |  |
| Industry_[T.Model] | -0.0240 | 0.0140 | -1.72 | 0.0866 | * |
| Industry_[T.Singer] | 0.0072 | 0.0114 | 0.64 | 0.5257 |  |
| Industry_[T.SocialMedia] | -0.0199 | 0.0177 | -1.12 | 0.2615 |  |
| Industry_[T.TV] | -0.0041 | 0.0099 | -0.41 | 0.6786 |  |
| age | -0.0019 | 0.0009 | -2.27 | 0.0234 | ** |
| age_squared | 0.0000 | 0.0000 | 0.76 | 0.4505 |  |
| is_us | -0.0057 | 0.0067 | -0.85 | 0.3955 |  |
| celeb_popularity | 0.0023 | 0.0007 | 3.29 | 0.0011 | *** |
| partner_popularity | -0.0022 | 0.0008 | -2.91 | 0.0038 | *** |
| bmi | -0.0002 | 0.0012 | -0.16 | 0.8753 |  |
| dance_experience_score | -0.0000 | 0.0013 | -0.02 | 0.9804 |  |

## Model 3: Final Placement (Lower is Better)

**R-squared:** 0.2411
**Adjusted R-squared:** 0.2150
**F-statistic:** 9.21 (p = 1.1003e-17)
**AIC:** 2222.4

| Variable | Coefficient | Std Error | t-value | p-value | Sig |
|----------|-------------|-----------|---------|---------|-----|
| Intercept | 0.4510 | 2.4188 | 0.19 | 0.8522 |  |
| Industry_[T.Actor] | -0.7680 | 0.6983 | -1.10 | 0.2720 |  |
| Industry_[T.Athlete] | -0.1545 | 0.7087 | -0.22 | 0.8275 |  |
| Industry_[T.Comedian] | 0.1876 | 1.1372 | 0.16 | 0.8691 |  |
| Industry_[T.Model] | 2.8206 | 1.0283 | 2.74 | 0.0064 | *** |
| Industry_[T.Singer] | -0.5813 | 0.8392 | -0.69 | 0.4889 |  |
| Industry_[T.SocialMedia] | 0.3116 | 1.2998 | 0.24 | 0.8106 |  |
| Industry_[T.TV] | 0.1347 | 0.7304 | 0.18 | 0.8538 |  |
| age | 0.2237 | 0.0626 | 3.57 | 0.0004 | *** |
| age_squared | -0.0012 | 0.0007 | -1.67 | 0.0956 | * |
| is_us | 0.2608 | 0.4922 | 0.53 | 0.5965 |  |
| celeb_popularity | -0.1408 | 0.0512 | -2.75 | 0.0062 | *** |
| partner_popularity | 0.1087 | 0.0568 | 1.91 | 0.0565 | * |
| bmi | -0.0299 | 0.0867 | -0.34 | 0.7304 |  |
| dance_experience_score | 0.1738 | 0.0975 | 1.78 | 0.0752 | * |

---
# Module A-2: Weekly-Level Analysis (Dynamic Social Media Effects)

## Model Specification

**周级别模型公式:**

$$Y_{it} = \beta_0 + \beta_1 \cdot Age_i + \beta_2 \cdot isUS_i + \beta_3 \cdot BMI_i + \beta_4 \cdot DanceExp_i + \sum_k \gamma_k \cdot Industry_k + \lambda_1 \cdot SocialPop_{it} + \lambda_2 \cdot GoogleSearch_{it} + \epsilon_{it}$$

> **注意**: 周级别模型样本量更大（N≈2777），静态变量的标准误更小，但系数含义与选手级模型相同。


**Sample Size:** 2777 week-contestant observations


## Weekly Model: Fan Vote Share

**R-squared:** 0.3195
**Adjusted R-squared:** 0.3160
**F-statistic:** 92.61 (p = 1.5505e-218)
**AIC:** -7914.6

| Variable | Coefficient | Std Error | t-value | p-value | Sig |
|----------|-------------|-----------|---------|---------|-----|
| Intercept | 0.1120 | 0.0147 | 7.61 | 0.0000 | *** |
| Industry_[T.Actor] | -0.0016 | 0.0049 | -0.32 | 0.7502 |  |
| Industry_[T.Athlete] | 0.0074 | 0.0050 | 1.47 | 0.1404 |  |
| Industry_[T.Comedian] | -0.0191 | 0.0085 | -2.25 | 0.0242 | ** |
| Industry_[T.Model] | 0.0090 | 0.0082 | 1.10 | 0.2726 |  |
| Industry_[T.Singer] | 0.0016 | 0.0059 | 0.28 | 0.7798 |  |
| Industry_[T.SocialMedia] | -0.0250 | 0.0079 | -3.17 | 0.0016 | *** |
| Industry_[T.TV] | -0.0079 | 0.0051 | -1.56 | 0.1197 |  |
| age | -0.0002 | 0.0001 | -1.82 | 0.0692 | * |
| is_us | 0.0011 | 0.0033 | 0.33 | 0.7447 |  |
| celeb_popularity | 0.0002 | 0.0002 | 0.97 | 0.3337 |  |
| bmi | 0.0001 | 0.0006 | 0.19 | 0.8503 |  |
| dance_experience_score | 0.0015 | 0.0007 | 2.30 | 0.0214 | ** |
| social_media_popularity | 0.0136 | 0.0004 | 30.92 | 0.0000 | *** |
| google_search_volume | 0.0001 | 0.0000 | 3.44 | 0.0006 | *** |

**Dynamic Effects:**
- Social Media Popularity: β = 0.0136 (p = 0.0000) ✓
- Google Search Volume: β = 0.0001 (p = 0.0006) ✓

## Weekly Model: Judge Score

**R-squared:** 0.3309
**Adjusted R-squared:** 0.3275
**F-statistic:** 97.56 (p = 1.3616e-228)
**AIC:** 17039.4

| Variable | Coefficient | Std Error | t-value | p-value | Sig |
|----------|-------------|-----------|---------|---------|-----|
| Intercept | 26.7148 | 1.3156 | 20.31 | 0.0000 | *** |
| Industry_[T.Actor] | 1.0153 | 0.4414 | 2.30 | 0.0215 | ** |
| Industry_[T.Athlete] | 0.5678 | 0.4457 | 1.27 | 0.2028 |  |
| Industry_[T.Comedian] | 1.1456 | 0.7560 | 1.52 | 0.1298 |  |
| Industry_[T.Model] | -0.1400 | 0.7312 | -0.19 | 0.8482 |  |
| Industry_[T.Singer] | 0.7030 | 0.5235 | 1.34 | 0.1794 |  |
| Industry_[T.SocialMedia] | 5.0674 | 0.7049 | 7.19 | 0.0000 | *** |
| Industry_[T.TV] | -0.3370 | 0.4547 | -0.74 | 0.4587 |  |
| age | -0.0692 | 0.0087 | -7.95 | 0.0000 | *** |
| is_us | 0.5337 | 0.2988 | 1.79 | 0.0742 | * |
| celeb_popularity | -0.0354 | 0.0170 | -2.08 | 0.0373 | ** |
| bmi | 0.0346 | 0.0532 | 0.65 | 0.5148 |  |
| dance_experience_score | -0.0672 | 0.0593 | -1.13 | 0.2573 |  |
| social_media_popularity | 1.1159 | 0.0394 | 28.33 | 0.0000 | *** |
| google_search_volume | 0.0064 | 0.0033 | 1.96 | 0.0503 | * |

---
# Module B: Professional Dancer Analysis (Mixed Effects Model)

## Model Specification

**混合效应模型:**

$$Y_{ij} = \beta_0 + X_{ij}\beta + u_j + \epsilon_{ij}$$

其中 $u_j \sim N(0, \sigma^2_u)$ 是舞者随机效应，ICC = $\sigma^2_u / (\sigma^2_u + \sigma^2_\epsilon)$


**Sample Size:** 421 contestant-partner pairs
**Unique Partners:** 60
**Partners with 2+ appearances:** 40

## Mixed Model 1: Average Judge Score

**Convergence:** Yes
**Log-Likelihood:** -1255.60
**Random Effect Variance (σ²_u):** 3.2006
**Residual Variance (σ²_ε):** 20.2293
**Intraclass Correlation (ICC):** 0.1366 (13.7%)

**Fixed Effects:**

| Variable | Coefficient | Std Error | z-value | p-value | Sig |
|----------|-------------|-----------|---------|---------|-----|
| Intercept | 26.1254 | 2.5786 | 10.13 | 0.0000 | *** |
| age | -0.1540 | 0.0176 | -8.77 | 0.0000 | *** |
| is_us | -0.0531 | 0.6703 | -0.08 | 0.9369 |  |
| celeb_popularity | 0.0595 | 0.0499 | 1.19 | 0.2332 |  |
| bmi | 0.0905 | 0.1091 | 0.83 | 0.4065 |  |
| dance_experience_score | 0.0875 | 0.1029 | 0.85 | 0.3952 |  |
| partner_seasons_before | 0.2054 | 0.0493 | 4.17 | 0.0000 | *** |

**Interpretation:** 13.7% of variance in judge score is attributable to differences between professional dancers.

## Mixed Model 2: Average Fan Vote Share

**Convergence:** Yes
**Log-Likelihood:** 668.23
**Random Effect Variance (σ²_u):** 0.0003
**Residual Variance (σ²_ε):** 0.0018
**Intraclass Correlation (ICC):** 0.1539 (15.4%)

**Fixed Effects:**

| Variable | Coefficient | Std Error | z-value | p-value | Sig |
|----------|-------------|-----------|---------|---------|-----|
| Intercept | 0.1469 | 0.0248 | 5.91 | 0.0000 | *** |
| age | -0.0012 | 0.0002 | -7.16 | 0.0000 | *** |
| is_us | -0.0001 | 0.0064 | -0.02 | 0.9847 |  |
| celeb_popularity | 0.0007 | 0.0005 | 1.44 | 0.1501 |  |
| bmi | 0.0002 | 0.0010 | 0.19 | 0.8464 |  |
| dance_experience_score | 0.0007 | 0.0010 | 0.67 | 0.5000 |  |
| partner_seasons_before | -0.0015 | 0.0005 | -2.98 | 0.0029 | *** |

**Interpretation:** 15.4% of variance in fan vote is attributable to differences between professional dancers.

## Mixed Model 3: Placement

**Convergence:** No
**Log-Likelihood:** -1119.32
**Random Effect Variance (σ²_u):** 2.2785
**Residual Variance (σ²_ε):** 10.2530
**Intraclass Correlation (ICC):** 0.1818 (18.2%)

**Fixed Effects:**

| Variable | Coefficient | Std Error | z-value | p-value | Sig |
|----------|-------------|-----------|---------|---------|-----|
| Intercept | 5.3718 | 1.8894 | 2.84 | 0.0045 | *** |
| age | 0.1103 | 0.0127 | 8.68 | 0.0000 | *** |
| is_us | -0.1106 | 0.4789 | -0.23 | 0.8174 |  |
| celeb_popularity | -0.0800 | 0.0389 | -2.05 | 0.0400 | ** |
| bmi | -0.1081 | 0.0791 | -1.37 | 0.1718 |  |
| dance_experience_score | 0.0619 | 0.0737 | 0.84 | 0.4009 |  |
| partner_seasons_before | 0.0404 | 0.0406 | 1.00 | 0.3196 |  |

**Interpretation:** 18.2% of variance in placement is attributable to differences between professional dancers.

## Partner Performance Ranking

Top 15 professional dancers by average partner placement (min 2 appearances):

| Rank | Partner | Apps | Avg Place | Best | Avg Judge | Avg Fan |
|------|---------|------|-----------|------|-----------|---------|
| 1 | Derek Hough | 17 | 2.94 | 1 | 28.7 | 0.145 |
| 2 | Julianne Hough | 5 | 4.20 | 1 | 23.6 | 0.132 |
| 3 | Daniella Karagach | 5 | 4.60 | 1 | 27.6 | 0.124 |
| 4 | Mark Ballas | 21 | 5.19 | 1 | 27.1 | 0.119 |
| 5 | Valentin Chmerkovskiy | 19 | 5.26 | 1 | 28.7 | 0.125 |
| 6 | Lindsay Arnold | 10 | 5.40 | 1 | 25.6 | 0.108 |
| 7 | Witney Carson | 14 | 5.50 | 1 | 28.1 | 0.103 |
| 8 | Cheryl Burke | 25 | 5.80 | 1 | 24.3 | 0.121 |
| 9 | Maksim Chmerkoskiy | 17 | 6.00 | 1 | 25.1 | 0.119 |
| 10 | Sasha Farber | 12 | 6.08 | 3 | 28.0 | 0.140 |
| 11 | Alan Bersten | 9 | 6.11 | 1 | 25.8 | 0.094 |
| 12 | Lacey Schwimmer | 6 | 6.17 | 2 | 20.6 | 0.112 |
| 13 | Kym Johnson | 14 | 6.29 | 1 | 22.8 | 0.115 |
| 14 | Jenna Johnson | 9 | 6.44 | 1 | 24.0 | 0.099 |
| 15 | Sharna Burgess | 14 | 6.57 | 1 | 26.0 | 0.107 |

---
# Module C: Effect Comparison - Judge Scores vs Fan Votes

## Coefficient Comparison (OLS Models)

| Variable | Judge t-stat | Fan t-stat | Δ | Who Cares More? |
|----------|--------------|------------|---|-----------------|
| age | -3.06 | -2.27 | -0.79 | **Judges** |
| is_us | -0.44 | -0.85 | +0.41 | Similar |
| celeb_popularity | -0.74 | 3.29 | +2.55 | **Fans** |
| bmi | 0.38 | -0.16 | -0.22 | Similar |
| dance_experience_score | -1.17 | -0.02 | -1.15 | **Judges** |

*Note: t-statistics indicate effect strength; larger absolute values = stronger effects.*

## Partner Effect Comparison (ICC from Mixed Models)

| Metric | Judge Model | Fan Model | Interpretation |
|--------|-------------|-----------|----------------|
| ICC | 13.7% | 15.4% | Partner → Fans |

**Key Finding:** Partner choice explains **15.4%** of fan vote variance vs **13.7%** for judges. Fans may be influenced by partner popularity.

---
# Key Findings Summary

## Finding 1: Age Effect
- **Monotonic effect**: younger contestants score higher
- Age coefficient: β = -0.2770 (p = 0.0024)

## Finding 2: Dance Experience Effect
- Not statistically significant (p = 0.2427)

## Finding 3: Popularity Effect
- Effect on judge scores: t = -0.74
- Effect on fan votes: t = 3.29
- **Fans are more influenced by celebrity popularity than judges**

## Finding 4: Industry Effect
Industry effects on judge scores (vs Other baseline):

| Industry | Coefficient | p-value | Interpretation |
|----------|-------------|---------|----------------|
| Actor | +1.84 | 0.0689 | Higher scores  |
| Athlete | +0.52 | 0.6133 | Higher scores  |
| Comedian | -0.35 | 0.8321 | Lower scores  |
| Model | -2.44 | 0.1010 | Lower scores  |
| Singer | +1.36 | 0.2636 | Higher scores  |
| SocialMedia | +5.30 | 0.0051 | Higher scores ✓ |
| TV | -0.37 | 0.7243 | Lower scores  |

## Finding 5: Social Media Dynamic Effect (Weekly Level)
- Social Media Popularity → Fan Vote: β = 0.0136 (p = 0.0000)
- Google Search Volume → Fan Vote: β = 0.0001 (p = 0.0006)
- **Weekly social media buzz significantly impacts fan voting**

## Finding 6: Professional Dancer Effect
- Partner explains **13.7%** of variance in judge scores (ICC)
- This is a **moderate** partner effect