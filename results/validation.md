# Monte Carlo validation

1000 A/A + 1000 A/B experiments, 40,000 users each, pre-period 28 d, experiment 14 d. Bootstrap: first 150 runs.

| method                     |   runs |   FPR (A/A), % |   power (+8 % conv.), % |
|:---------------------------|-------:|---------------:|------------------------:|
| GMV: Welch t-test          |   1000 |            6.0 |                    66.6 |
| GMV: Mann-Whitney          |   1000 |            6.3 |                    79.6 |
| GMV: CUPED                 |   1000 |            6.4 |                    76.9 |
| GMV: CUPED, 3 covariates   |   1000 |            5.9 |                    78.5 |
| ratio: naive session-level |   1000 |           13.9 |                    98.5 |
| ratio: delta method        |   1000 |            5.9 |                    95.5 |
| GMV: Poisson bootstrap     |    150 |            6.7 |                    68.0 |

Acceptable FPR band for 1000 runs: 5.0 ± 1.4 %.

Mean variance reduction: CUPED rho^2 = 0.239; 3 covariates R^2 = 0.270 -> sample size needed for the same power is 27 % smaller.
Power of GMV test: 67 % -> 78 % with regression-adjusted CUPED.