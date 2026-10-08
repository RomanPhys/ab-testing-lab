"""Fast checks of the statistical machinery.  Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ablab import stats as S  # noqa: E402
from ablab.population import hash_split, sample_population, simulate_period  # noqa: E402


def test_sample_size_and_mde_are_inverse():
    n = S.sample_size(sigma=10.0, mde_abs=0.5)
    assert abs(S.mde(10.0, n) - 0.5) < 0.01


def test_srm_detects_broken_split():
    assert S.srm_test(50_000, 50_000) > 0.5
    assert S.srm_test(50_600, 49_400) < 1e-3   # 50.6 / 49.4 on 100k users is already broken


def test_hash_split_is_deterministic_and_balanced():
    ids = np.arange(20_000)
    a, b = hash_split(ids, "exp1"), hash_split(ids, "exp1")
    assert (a == b).all() and abs(a.mean() - 0.5) < 0.02
    assert abs(np.corrcoef(a, hash_split(ids, "exp2"))[0, 1]) < 0.03  # salts are independent


def test_delta_method_matches_bootstrap():
    rng = np.random.default_rng(0)
    pop = sample_population(20_000, rng)
    out = simulate_period(pop, 14, rng)
    num, den = out.orders.astype(float), out.sessions.astype(float)
    _, var = S._delta_var(num, den)
    idx = rng.integers(0, num.size, size=(400, num.size))
    boot = num[idx].sum(1) / den[idx].sum(1)
    assert abs(np.sqrt(var) / boot.std() - 1) < 0.15


def test_cuped_reduces_variance_and_keeps_zero_effect():
    rng = np.random.default_rng(1)
    effects, ses_plain, ses_cuped = [], [], []
    for _ in range(60):
        pop = sample_population(10_000, rng)
        pre, post = simulate_period(pop, 28, rng), simulate_period(pop, 14, rng)
        g = rng.permutation(10_000) < 5_000
        r_plain = S.welch_ttest(post.gmv[~g], post.gmv[g])
        r_cup, _ = S.cuped(post.gmv[~g], post.gmv[g], pre.gmv[~g], pre.gmv[g])
        effects.append(r_cup.effect / r_cup.se)
        ses_plain.append(r_plain.se); ses_cuped.append(r_cup.se)
    assert abs(np.mean(effects)) < 0.4                 # no bias under H0
    assert np.mean(ses_cuped) < 0.95 * np.mean(ses_plain)
