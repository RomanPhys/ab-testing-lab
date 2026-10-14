"""
Statistical tests for A/B experiments, written out by hand so the formulas are easy to check.

Every function returns a TestResult with the absolute difference (treatment - control),
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


# means
def welch_ttest(x_c: np.ndarray, x_t: np.ndarray) -> TestResult:
    """Difference of means with unequal variances. At n ~ 1e4 t and z coincide, so z is used."""
    effect = x_t.mean() - x_c.mean()
    se = np.sqrt(x_t.var(ddof=1) / x_t.size + x_c.var(ddof=1) / x_c.size)
    return _from_z(effect, se)


def cuped(y_c: np.ndarray, y_t: np.ndarray, x_c: np.ndarray, x_t: np.ndarray) -> tuple[TestResult, float]:
    """
    CUPED (Deng et al., 2013), basically control variates from Monte Carlo.
        y_cv = y - theta (x - E x),   theta = Cov(y, x) / Var(x)
    x is measured before the experiment, so it does not depend on the assignment and the estimate
    stays unbiased. Var(y_cv) = Var(y) (1 - rho^2).
    theta is fitted on both groups together. Separate thetas per group would add bias.
    Returns the test result and rho^2.
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
    CUPED with several pre-period covariates, i.e. regression adjustment (Lin, 2013).
        y_adj = y - (X - mean X) beta,   beta from OLS of y on X over both groups
    Variance goes down by R^2 of this regression. With one covariate this is the same as cuped().
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
    """Rank test. It checks stochastic dominance, not a difference of means, here only for comparison."""
    p = st.mannwhitneyu(x_t, x_c, alternative="two-sided").pvalue
    return TestResult(np.median(x_t) - np.median(x_c), np.nan, np.nan, np.nan, p)


def poisson_bootstrap(x_c: np.ndarray, x_t: np.ndarray, n_boot: int = 1000,
                      rng: np.random.Generator | None = None) -> TestResult:
    """
    Each observation gets a Poisson(1) weight instead of multinomial resampling. Asymptotically
    it is the same, but replicas are independent weighted means that vectorise well, and it can be
    computed in a streaming or distributed way. Experiment platforms use it for that reason.
    """
    rng = rng or np.random.default_rng()
    w_c = rng.poisson(1.0, size=(n_boot, x_c.size))
    w_t = rng.poisson(1.0, size=(n_boot, x_t.size))
    diff = (w_t @ x_t) / w_t.sum(1) - (w_c @ x_c) / w_c.sum(1)
    effect = x_t.mean() - x_c.mean()
    lo, hi = np.quantile(diff, [0.025, 0.975])
    # two-sided p-value from the share of replicas on the other side of zero
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    return TestResult(effect, diff.std(ddof=1), lo, hi, max(p, 1.0 / n_boot))


# ratio metrics sum(num) / sum(den), e.g. orders per session or GMV per order
def ratio_naive(num_c, den_c, num_t, den_t) -> TestResult:
    """
    Wrong on purpose, every session is treated as an independent Bernoulli trial (two-proportion z-test).
    Sessions of one user share p_u, they are positively correlated and the real variance is larger.
    """
    r_c, r_t = num_c.sum() / den_c.sum(), num_t.sum() / den_t.sum()
    se = np.sqrt(r_c * (1 - r_c) / den_c.sum() + r_t * (1 - r_t) / den_t.sum())
    return _from_z(r_t - r_c, se)


def _delta_var(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """
    Delta method, i.e. first-order error propagation for R = mean(N) / mean(D).
        Var(R) ~ (1/mu_D^2) (Var N - 2 R Cov(N, D) + R^2 Var D) / n
    Users are randomised, so the (N_u, D_u) pairs are i.i.d. even though sessions are not.
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


# sanity checks and design
def srm_test(n_c: int, n_t: int, share_t: float = 0.5) -> float:
    """Sample Ratio Mismatch, chi-square test of the observed split. p < 0.001 means the split is broken."""
    n = n_c + n_t
    return st.chisquare([n_c, n_t], [n * (1 - share_t), n * share_t]).pvalue


def sample_size(sigma: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per group, n = 2 sigma^2 (z_{1-a/2} + z_{power})^2 / MDE^2."""
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return int(np.ceil(2 * sigma ** 2 * z ** 2 / mde_abs ** 2))


def mde(sigma: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return float(z * sigma * np.sqrt(2 / n_per_group))
