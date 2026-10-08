"""
Statistical methods for A/B tests, implemented explicitly so that every formula is visible.

All functions return a TestResult: point estimate of the absolute difference (treatment - control),
its standard error, a 95 % CI and a two-sided p-value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as st

Z975 = st.norm.ppf(0.975)


@dataclass
class TestResult:
    effect: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float


def _from_z(effect: float, se: float) -> TestResult:
    z = effect / se if se > 0 else 0.0
    return TestResult(effect, se, effect - Z975 * se, effect + Z975 * se, 2 * st.norm.sf(abs(z)))


# --------------------------------------------------------------------------- #
#  Means
# --------------------------------------------------------------------------- #
def welch_ttest(x_c: np.ndarray, x_t: np.ndarray) -> TestResult:
    """Difference of means with unequal variances. With n ~ 1e4 the CLT makes z == t."""
    effect = x_t.mean() - x_c.mean()
    se = np.sqrt(x_t.var(ddof=1) / x_t.size + x_c.var(ddof=1) / x_c.size)
    return _from_z(effect, se)


def cuped(y_c: np.ndarray, y_t: np.ndarray, x_c: np.ndarray, x_t: np.ndarray) -> tuple[TestResult, float]:
    """
    CUPED (Deng et al., 2013) = control variates from Monte Carlo.
        y_cv = y - theta (x - E x),   theta = Cov(y, x) / Var(x)
    x is a pre-experiment covariate, so it is independent of the assignment and the estimator stays
    unbiased, while Var(y_cv) = Var(y) (1 - rho^2). Returns the test and the variance reduction rho^2.
    theta is estimated on the pooled sample (same theta for both groups, otherwise bias creeps in).
    """
    y = np.concatenate([y_c, y_t])
    x = np.concatenate([x_c, x_t])
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    mx = x.mean()
    res = welch_ttest(y_c - theta * (x_c - mx), y_t - theta * (x_t - mx))
    rho2 = np.corrcoef(y, x)[0, 1] ** 2
    return res, rho2


def cuped_multi(y_c: np.ndarray, y_t: np.ndarray, X_c: np.ndarray, X_t: np.ndarray) -> tuple[TestResult, float]:
    """
    CUPED with several pre-period covariates = regression adjustment (Lin, 2013):
        y_adj = y - (X - mean X) beta,   beta = OLS of y on X on the pooled sample.
    Geometrically: project y onto the span of the covariates and keep only the orthogonal residual.
    Variance reduction is R^2 of that regression; with one covariate it reduces to rho^2.
    """
    y = np.concatenate([y_c, y_t])
    X = np.vstack([X_c, X_t])
    Xc = X - X.mean(0)
    beta, *_ = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)
    adj = Xc @ beta
    r2 = 1 - np.var(y - adj) / np.var(y)
    n_c = y_c.size
    return welch_ttest(y_c - adj[:n_c], y_t - adj[n_c:]), float(r2)


def mann_whitney(x_c: np.ndarray, x_t: np.ndarray) -> TestResult:
    """Rank test: tests stochastic dominance, NOT a difference of means. Kept for comparison."""
    p = st.mannwhitneyu(x_t, x_c, alternative="two-sided").pvalue
    return TestResult(np.median(x_t) - np.median(x_c), np.nan, np.nan, np.nan, p)


def poisson_bootstrap(x_c: np.ndarray, x_t: np.ndarray, n_boot: int = 1000,
                      rng: np.random.Generator | None = None) -> TestResult:
    """
    Each observation gets a Poisson(1) weight instead of multinomial resampling: the same
    asymptotics, but every replica is an independent vectorised weighted mean (and it can be
    computed in a streaming / distributed way, which is why platforms use it).
    """
    rng = rng or np.random.default_rng()
    w_c = rng.poisson(1.0, size=(n_boot, x_c.size))
    w_t = rng.poisson(1.0, size=(n_boot, x_t.size))
    diff = (w_t @ x_t) / w_t.sum(1) - (w_c @ x_c) / w_c.sum(1)
    effect = x_t.mean() - x_c.mean()
    lo, hi = np.quantile(diff, [0.025, 0.975])
    # two-sided p-value: how often the bootstrap distribution crosses zero
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    return TestResult(effect, diff.std(ddof=1), lo, hi, max(p, 1.0 / n_boot))


# --------------------------------------------------------------------------- #
#  Ratio metrics: sum(num) / sum(den), e.g. orders per session, GMV per order
# --------------------------------------------------------------------------- #
def ratio_naive(num_c, den_c, num_t, den_t) -> TestResult:
    """
    WRONG on purpose: treats every session as an independent Bernoulli trial (two-proportion z-test).
    Sessions of one user share p_u, so they are positively correlated and the true variance is larger.
    """
    r_c, r_t = num_c.sum() / den_c.sum(), num_t.sum() / den_t.sum()
    se = np.sqrt(r_c * (1 - r_c) / den_c.sum() + r_t * (1 - r_t) / den_t.sum())
    return _from_z(r_t - r_c, se)


def _delta_var(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """
    Delta method = first-order error propagation for R = mean(N) / mean(D):
        Var(R) ~ (1/mu_D^2) (Var N - 2 R Cov(N, D) + R^2 Var D) / n
    The randomisation unit is the user, so (N_u, D_u) pairs are i.i.d. even if sessions are not.
    """
    n = num.size
    mu_n, mu_d = num.mean(), den.mean()
    r = mu_n / mu_d
    c = np.cov(num, den)
    var = (c[0, 0] - 2 * r * c[0, 1] + r ** 2 * c[1, 1]) / (mu_d ** 2 * n)
    return r, var


def ratio_delta(num_c, den_c, num_t, den_t) -> TestResult:
    r_c, v_c = _delta_var(num_c, den_c)
    r_t, v_t = _delta_var(num_t, den_t)
    return _from_z(r_t - r_c, np.sqrt(v_c + v_t))


# --------------------------------------------------------------------------- #
#  Sanity checks and design
# --------------------------------------------------------------------------- #
def srm_test(n_c: int, n_t: int, share_t: float = 0.5) -> float:
    """Sample Ratio Mismatch: chi-square goodness of fit of the observed split. p < 0.001 -> broken."""
    n = n_c + n_t
    return st.chisquare([n_c, n_t], [n * (1 - share_t), n * share_t]).pvalue


def sample_size(sigma: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per group:  n = 2 sigma^2 (z_{1-a/2} + z_{power})^2 / MDE^2."""
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return int(np.ceil(2 * sigma ** 2 * z ** 2 / mde_abs ** 2))


def mde(sigma: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return float(z * sigma * np.sqrt(2 / n_per_group))
