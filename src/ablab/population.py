"""
Generative model of a marketplace population.

Each user u has latent traits that persist over time (this is what makes pre-period data
useful for variance reduction):
    lambda_u ~ Gamma            sessions per day          (activity, heavy right tail)
    p_u      ~ Beta             P(order | session)        (propensity to buy)
    m_u      ~ Normal           log of typical order value

Observed in a period of D days:
    sessions_u ~ Poisson(lambda_u * D)
    orders_u   ~ Binomial(sessions_u, p_u * (1 + uplift * treated_u))
    gmv_u      = sum of orders_u draws from LogNormal(m_u, VALUE_SIGMA)

Heterogeneity across users is the key: sessions of one user are NOT independent trials,
because they share the same p_u. Any test that treats sessions as i.i.d. underestimates variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALUE_SIGMA = 0.8
PLATFORMS = np.array(["android", "ios", "web"])
REGIONS = np.array(["moscow", "spb", "regions"])


@dataclass
class Population:
    lam: np.ndarray
    p: np.ndarray
    m: np.ndarray
    platform: np.ndarray
    region: np.ndarray

    @property
    def n(self) -> int:
        return self.lam.size


def sample_population(n_users: int, rng: np.random.Generator) -> Population:
    lam = rng.gamma(shape=0.7, scale=0.8 / 0.7, size=n_users)     # mean 0.8 sessions/day
    p = rng.beta(1.2, 35.0, size=n_users)                          # mean ~3.3 %
    m = rng.normal(np.log(1500.0), 0.5, size=n_users)              # typical check ~1500 RUB
    platform = rng.choice(PLATFORMS, size=n_users, p=[0.5, 0.3, 0.2])
    region = rng.choice(REGIONS, size=n_users, p=[0.35, 0.15, 0.5])
    return Population(lam, p, m, platform, region)


@dataclass
class PeriodOutcome:
    sessions: np.ndarray
    orders: np.ndarray
    gmv: np.ndarray


def simulate_period(pop: Population, days: int, rng: np.random.Generator,
                    treated: np.ndarray | None = None, uplift: float = 0.0) -> PeriodOutcome:
    """User-level aggregates for one period. Exact sampling, vectorised (no per-user loop)."""
    p = pop.p if treated is None else pop.p * (1.0 + uplift * treated)
    sessions = rng.poisson(pop.lam * days)
    orders = rng.binomial(sessions, np.clip(p, 0, 1))
    owner = np.repeat(np.arange(pop.n), orders)                    # one entry per order
    values = rng.lognormal(pop.m[owner], VALUE_SIGMA)
    gmv = np.bincount(owner, weights=values, minlength=pop.n)
    return PeriodOutcome(sessions, orders, gmv)


def hash_split(user_ids: np.ndarray, salt: str, share: float = 0.5) -> np.ndarray:
    """
    Deterministic assignment as on real experiment platforms: variant = hash(salt + id) mod 1000.
    Same user always lands in the same group of an experiment; different salts are independent.
    """
    import hashlib
    buckets = np.fromiter(
        (int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest()[:8], 16) % 1000 for u in user_ids),
        dtype=np.int64, count=len(user_ids))
    return (buckets < share * 1000).astype(np.int8)
