# Case study: `new_checkout_button`

Users: 60,000; pre-period 28 d, experiment 14 d. True effect injected: +8 % to P(order | session).

## 0. Design

MDE for GMV per user (alpha = 0.05, power = 0.8, 30,000 users per group), estimated on a historical 14-day window: **7.6 %** with a plain t-test; variance reduction lowers it by a factor sqrt(1 - R^2) (see below).

## 1. Sample Ratio Mismatch

| variant   |   n_users |
|:----------|----------:|
| control   |     30001 |
| treatment |     29999 |

chi-square p = 0.993 -> split is healthy

## 2. Metrics

| metric             | method                                |   effect_abs |   effect_rel_% | ci95_rel_%      |   p_value |
|:-------------------|:--------------------------------------|-------------:|---------------:|:----------------|----------:|
| GMV per user       | Welch t-test                          |    30.64     |          3.417 | [-1.78, +8.62]  | 0.1978    |
| GMV per user       | CUPED (covariate: pre-period GMV)     |    54.16     |          6.04  | [+1.47, +10.61] | 0.009623  |
| GMV per user       | CUPED, 3 covariates (regression adj.) |    55.81     |          6.225 | [+1.78, +10.67] | 0.006094  |
| GMV per user       | Poisson bootstrap                     |    30.64     |          3.417 | [-1.57, +8.85]  | 0.188     |
| Buyer conversion   | Welch t-test                          |     0.008982 |          3.955 | [+0.98, +6.93]  | 0.00911   |
| Orders per session | Delta method                          |     0.002236 |          6.596 | [+3.16, +10.03] | 0.0001705 |
| Orders per session | Naive session-level z-test (wrong)    |     0.002236 |          6.596 | [+3.99, +9.20]  | 6.689e-07 |

Variance reduction: CUPED rho^2 = 0.227, 3 covariates R^2 = 0.268 -> the same power with 27 % fewer users.

## 3. Segments (CUPED, Holm-corrected)

| platform   |   n_users |   effect_rel_% |   p_value |   p_holm |
|:-----------|----------:|---------------:|----------:|---------:|
| android    |     30040 |         6.134  |   0.05361 |  0.1072  |
| ios        |     18044 |        10.71   |   0.01079 |  0.03238 |
| web        |     11916 |         0.4015 |   0.9371  |  0.9371  |

## 4. Decision

Decide on the pre-registered primary metric (GMV per user, CUPED with 3 covariates). Segment differences are exploratory unless they survive the correction.