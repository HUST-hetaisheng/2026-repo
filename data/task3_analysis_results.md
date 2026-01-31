# Task 3 Analysis Results
# Impact of Contestant and Partner Characteristics

*Generated: 2026-01-31 20:20*

---

## Module A: Celebrity Characteristics Analysis (OLS)

### Model Specification

$$Y_i = \beta_0 + \beta_1 \cdot Age_i + \beta_2 \cdot Age_i^2 + \beta_3 \cdot isUS_i + \beta_4 \cdot \log(Popularity_i + 1) + \epsilon_i$$


**Sample Size:** 421 contestants with complete data


### Model 1: Average Judge Score

**R-squared:** 0.1928
**Adjusted R-squared:** 0.1831
**F-statistic:** 19.83 (p = 0.0000)

| Variable | Coefficient | Std Error | t-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
| Intercept | 33.7901 | 2.0314 | 16.63 | 0.0000 | *** |
| age | -0.3428 | 0.0892 | -3.84 | 0.0001 | *** |
| age_squared | 0.0020 | 0.0010 | 2.01 | 0.0451 | ** |
| is_us | 0.1148 | 0.7057 | 0.16 | 0.8708 |  |
| celeb_pop_log | -0.7185 | 0.6085 | -1.18 | 0.2384 |  |
| partner_pop_log | 0.8607 | 0.6125 | 1.41 | 0.1607 |  |

### Model 2: Average Fan Vote Share

**R-squared:** 0.1475
**Adjusted R-squared:** 0.1373
**F-statistic:** 14.36 (p = 0.0000)

| Variable | Coefficient | Std Error | t-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
| Intercept | 0.1671 | 0.0189 | 8.85 | 0.0000 | *** |
| age | -0.0018 | 0.0008 | -2.20 | 0.0286 | ** |
| age_squared | 0.0000 | 0.0000 | 0.68 | 0.4942 |  |
| is_us | -0.0019 | 0.0066 | -0.28 | 0.7764 |  |
| celeb_pop_log | 0.0161 | 0.0057 | 2.85 | 0.0046 | *** |
| partner_pop_log | -0.0164 | 0.0057 | -2.88 | 0.0042 | *** |

### Model 3: Final Placement (Lower is Better)

**R-squared:** 0.2070
**Adjusted R-squared:** 0.1975
**F-statistic:** 21.67 (p = 0.0000)

| Variable | Coefficient | Std Error | t-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
| Intercept | 0.5591 | 1.3961 | 0.40 | 0.6890 |  |
| age | 0.2264 | 0.0613 | 3.69 | 0.0003 | *** |
| age_squared | -0.0013 | 0.0007 | -1.83 | 0.0683 | * |
| is_us | -0.0809 | 0.4850 | -0.17 | 0.8675 |  |
| celeb_pop_log | -1.0936 | 0.4182 | -2.62 | 0.0092 | *** |
| partner_pop_log | 0.9245 | 0.4210 | 2.20 | 0.0286 | ** |

---
## Module B: Professional Dancer Analysis (Mixed Effects Model)

### Model Specification

$$Y_{ij} = \beta_0 + \beta_1 \cdot Age_{ij} + \beta_2 \cdot isUS_{ij} + \beta_3 \cdot \log(Pop_{ij}+1) + \beta_4 \cdot PartnerExp_j + u_j + \epsilon_{ij}$$

Where:
- $i$ indexes contestants, $j$ indexes professional dancers
- $u_j \sim N(0, \sigma^2_u)$ is the random effect for dancer $j$
- $\sigma^2_u / (\sigma^2_u + \sigma^2_\epsilon)$ is the Intraclass Correlation (ICC)


**Sample Size:** 421 contestant-partner pairs
**Unique Partners:** 60
**Partners with 2+ appearances:** 40

### Mixed Model 1: Average Judge Score

**Convergence:** Yes
**Log-Likelihood:** -1252.09
**Random Effect Variance (σ²_u):** 3.2014
**Residual Variance (σ²_ε):** 20.2299
**Intraclass Correlation (ICC):** 0.1366 (13.7%)

**Fixed Effects:**

| Variable | Coefficient | Std Error | z-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
| Intercept | 28.2868 | 1.1182 | 25.30 | 0.0000 | *** |
| age | -0.1531 | 0.0175 | -8.75 | 0.0000 | *** |
| is_us | -0.0443 | 0.6695 | -0.07 | 0.9473 |  |
| celeb_pop_log | 0.3289 | 0.3196 | 1.03 | 0.3036 |  |
| partner_seasons_before | 0.1963 | 0.0485 | 4.05 | 0.0001 | *** |

**Interpretation:** 13.7% of the variance in judge scores is attributable to differences between professional dancers (partner effect).

### Mixed Model 2: Average Fan Vote Share

**Convergence:** Yes
**Log-Likelihood:** 680.75
**Random Effect Variance (σ²_u):** 0.0003
**Residual Variance (σ²_ε):** 0.0018
**Intraclass Correlation (ICC):** 0.1526 (15.3%)

**Fixed Effects:**

| Variable | Coefficient | Std Error | z-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
| Intercept | 0.1580 | 0.0109 | 14.47 | 0.0000 | *** |
| age | -0.0012 | 0.0002 | -7.22 | 0.0000 | *** |
| is_us | -0.0001 | 0.0064 | -0.01 | 0.9927 |  |
| celeb_pop_log | 0.0008 | 0.0032 | 0.25 | 0.8001 |  |
| partner_seasons_before | -0.0017 | 0.0005 | -3.34 | 0.0008 | *** |

**Interpretation:** 15.3% of the variance in fan votes is attributable to differences between professional dancers.

### Partner Performance Ranking

Top 10 professional dancers by average partner placement:

| Rank | Partner | Appearances | Avg Placement | Best Placement | Avg Judge | Avg Fan Vote |
|------|---------|-------------|---------------|----------------|-----------|--------------|
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

---
## Module C: Effect Comparison - Judge Scores vs Fan Votes

### Key Question: Do characteristics impact judges and fans differently?

### Standardized Coefficient Comparison (OLS Models)

| Variable | Judge Score (β) | Fan Vote (β) | Difference | Interpretation |
|----------|-----------------|--------------|------------|----------------|
| age | -3.84 | -2.20 | 1.64 | Age: Higher judge impact |
| is_us | 0.16 | -0.28 | -0.12 | US Origin: Higher judge impact |
| celeb_pop_log | -1.18 | 2.85 | -1.67 | Popularity: Higher fan impact |
| partner_pop_log | 1.41 | -2.88 | -1.48 |  |

*Note: Values are t-statistics; larger absolute values indicate stronger effects.*

### Partner Effect Comparison (ICC from Mixed Models)

| Metric | Judge Score Model | Fan Vote Model | Interpretation |
|--------|-------------------|----------------|----------------|
| ICC | 0.1366 (13.7%) | 0.1526 (15.3%) | Partner matters more for fans |
| σ²_partner | 3.2014 | 0.0003 | Variance from partner differences |
| σ²_residual | 20.2299 | 0.0018 | Unexplained variance |

**Key Finding:** Professional dancer choice explains **15.3%** of fan vote variance vs **13.7%** for judges. This suggests fans are influenced by partner popularity/familiarity.

### Correlation: Judge Scores vs Fan Votes

- **Pearson Correlation:** r = 0.5248
- **p-value:** 3.6444e-31
- **Sample Size:** n = 421

**Interpretation:** Moderate correlation (r = 0.52) suggests some agreement between judges and fans, but also systematic differences in what they value.

---
## Key Findings Summary

### Finding 1: Age Effect
- Age coefficient: β₁ = -0.3428 (p = 0.0001)
- Age² coefficient: β₂ = 0.002012 (p = 0.0451)
- Age effect is **monotonic** (optimal age = 85 outside typical range)
- **Younger contestants tend to receive higher judge scores**

### Finding 2: Popularity Effect
- Celebrity popularity effect on judge scores: t = -1.18
- Celebrity popularity effect on fan votes: t = 2.85
- **Fans are more influenced by celebrity popularity than judges**

### Finding 3: US vs International Contestants
- No significant difference between US and international contestants (p = 0.7764)

### Finding 4: Professional Dancer Effect
- Professional dancer explains **13.7%** of variance in judge scores (ICC)
- This is a **moderate** partner effect

### Finding 5: Partner Experience
- Partner experience has a **positive** effect on performance (coef = 0.196, p = 0.0001)
- Each additional season of partner experience increases judge scores by 0.20 points on average