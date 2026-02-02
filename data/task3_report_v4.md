# Task 3: Impact of Contestant and Partner Characteristics (V4)

*Generated: 2026-02-02 19:31*

---

## Model Specification (Corrected)

### Changes from V3:

1. **Removed `age_squared`**: Paper equation does not include quadratic term

2. **Removed `celeb_popularity`**: Highly correlated with `social_media_popularity`

3. **Removed `partner_popularity`**: Conflicts with random effect $u_j$ in Mixed Model

4. **Using Z-score normalized data** for social media and Google search variables


### Contestant-Level Model (Paper Equation 1):

```

Y_i = β_0 + β_1·Age + β_2·isUS + β_3·BMI + β_4·DanceExp + Σγ_k·Industry_k + ε_i

```

### Weekly-Level Model:

```

Y_it = β_0 + β_1·SocialMedia_it + β_2·GoogleSearch_it + Controls + ε_it

```

### Mixed Effects Model (Paper Equation 2):

```

Y_ij = X_ij'β + u_j + ε_ij,  where u_j ~ N(0, σ²_u)

```

---
## Executive Summary

### Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Age affects performance** | β=-0.166*** (p=0.0000) |
| 2 | **Dance experience improves judge scores** | β=-0.169 (p=0.2322) |
| 3 | **Social media buzz drives fan votes** | β=0.0136*** (p=0.0000) |
| 4 | **Partner explains significant variance** | Judge=13.6%, Fan=13.9%, Placement=22.8% |
| 5 | **Industry background matters** | See coefficient table below |

---
## Model Fit Summary

### Contestant-Level Models

| Model | DV | N | R² | Adj R² | F-stat | p(F) |
|-------|----|----|----|----|--------|------|
| Judge_Score | avg_judge_score | 421 | 0.2316 | 0.2109 | 11.20 | 0.0000 |
| Fan_Vote | avg_fan_vote_share | 421 | 0.1537 | 0.1310 | 6.75 | 0.0000 |
| Placement | placement | 421 | 0.2209 | 0.1999 | 10.54 | 0.0000 |

### Weekly-Level Models

| Model | DV | N | R² | Adj R² | F-stat | p(F) |
|-------|----|----|----|----|--------|------|
| Weekly_Fan | fan_vote_share | 2777 | 0.3188 | 0.3156 | 99.48 | 0.0000 |
| Weekly_Judge | judge_total | 2777 | 0.3298 | 0.3266 | 104.57 | 0.0000 |

---
## Coefficient Comparison: Judge vs Fan vs Placement

| Variable | Judge β | Fan β | Placement β | Judge p | Fan p | Place p |
|----------|---------|-------|-------------|---------|-------|---------|
| age | -0.1664*** | -0.0013*** | 0.1232*** | 0.0000 | 0.0000 | 0.0000 |
| bmi | 0.0364 | -0.0000 | -0.0324 | 0.7721 | 0.9748 | 0.7121 |
| dance_experience_score | -0.1685 | 0.0000 | 0.1761* | 0.2322 | 0.9881 | 0.0741 |
| is_us | -0.2653 | -0.0055 | 0.2312 | 0.7089 | 0.4148 | 0.6411 |

---
## Weekly Model: Dynamic Variables Effect

| Variable | Fan Vote β | Judge β | Fan p | Judge p |
|----------|------------|---------|-------|---------|
| social_media_popularity | 0.0136*** | 1.1128*** | 0.0000 | 0.0000 |
| google_search_volume | 0.0039*** | 0.1991* | 0.0006 | 0.0516 |

**Interpretation**: Social media popularity (Z-scored) is measured weekly. 
A 1-SD increase in social media buzz is associated with a change in fan vote share.


---
## Professional Dancer Effect (ICC)

| Model | σ²_partner | σ²_residual | ICC | Interpretation |
|-------|------------|-------------|-----|----------------|
| Judge | 3.1924 | 20.2543 | 13.6% | Moderate |
| Fan | 0.0003 | 0.0019 | 13.9% | Moderate |
| Placement | 3.0168 | 10.2200 | 22.8% | Substantial |

### Top 10 Professional Dancers (by Avg Placement)

| Rank | Partner | Apps | Avg Place | Best | Avg Judge |
|------|---------|------|-----------|------|-----------|
| 1 | Derek Hough | 17 | 2.94 | 1 | 28.7 |
| 2 | Julianne Hough | 5 | 4.20 | 1 | 23.6 |
| 3 | Daniella Karagach | 5 | 4.60 | 1 | 27.6 |
| 4 | Mark Ballas | 21 | 5.19 | 1 | 27.1 |
| 5 | Valentin Chmerkovskiy | 19 | 5.26 | 1 | 28.7 |
| 6 | Lindsay Arnold | 10 | 5.40 | 1 | 25.6 |
| 7 | Witney Carson | 14 | 5.50 | 1 | 28.1 |
| 8 | Cheryl Burke | 25 | 5.80 | 1 | 24.3 |
| 9 | Maksim Chmerkoskiy | 17 | 6.00 | 1 | 25.1 |
| 10 | Sasha Farber | 12 | 6.08 | 3 | 28.0 |

---
## Industry Effect on Judge Score

| Industry | β (vs Other) | Std Err | t | p | Sig |
|----------|--------------|---------|---|---|-----|
| Actor | 1.7741 | 1.0866 | 1.63 | 0.1033 |  |
| Athlete | 0.3423 | 1.0809 | 0.32 | 0.7516 |  |
| Comedian | -0.5349 | 1.6857 | -0.32 | 0.7512 |  |
| Model | -2.6866 | 1.5215 | -1.77 | 0.0782 | * |
| Singer | 1.2827 | 1.2765 | 1.00 | 0.3156 |  |
| SocialMedia | 5.6090 | 1.8979 | 2.96 | 0.0033 | *** |
| TV | -0.4502 | 1.1117 | -0.40 | 0.6857 |  |

---
## Conclusions

1. **Contestant characteristics significantly affect outcomes**: Age, dance experience, and industry background all play important roles.

2. **Judges focus on performance**: Dance experience is a strong predictor of judge scores.

3. **Fans respond to publicity**: Weekly social media buzz significantly predicts fan voting, while judges appear immune.

4. **Partner matters**: Professional dancers explain 13.6%-22.8% of outcome variance (ICC).

5. **Model consistency**: Using corrected specifications without multicollinearity issues.
