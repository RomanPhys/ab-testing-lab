"""
Monte Carlo check of the statistical methods on data where the true effect is known.

  A/A runs (no effect): share of p < 0.05 should be about 5 %, p-values should be Uniform(0, 1).
  A/B runs (known effect): share of p < 0.05 is the power.

    python scripts/run_validation.py            # ~2.5 min
    python scripts/run_validation.py --runs 200 # quick run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ablab.population import sample_population, simulate_period  # noqa: E402
from ablab import stats as S  # noqa: E402

N_USERS, PRE_DAYS, EXP_DAYS, UPLIFT = 40_000, 28, 14, 0.08


def one_run(rng: np.random.Generator, uplift: float, with_bootstrap: bool) -> dict:
    pop = sample_population(N_USERS, rng)
    pre = simulate_period(pop, PRE_DAYS, rng)
    treated = rng.permutation(np.r_[np.zeros(N_USERS // 2), np.ones(N_USERS - N_USERS // 2)]).astype(int)
    post = simulate_period(pop, EXP_DAYS, rng, treated=treated, uplift=uplift)
    c, t = treated == 0, treated == 1
    X = np.c_[pre.gmv, pre.orders, pre.sessions]

    out = {
        "GMV: Welch t-test": S.welch_ttest(post.gmv[c], post.gmv[t]).pvalue,
        "GMV: Mann-Whitney": S.mann_whitney(post.gmv[c], post.gmv[t]).pvalue,
    }
    r, rho2 = S.cuped(post.gmv[c], post.gmv[t], pre.gmv[c], pre.gmv[t])
    out["GMV: CUPED"] = r.pvalue
    r, r2 = S.cuped_multi(post.gmv[c], post.gmv[t], X[c], X[t])
    out["GMV: CUPED, 3 covariates"] = r.pvalue
    out["ratio: naive session-level"] = S.ratio_naive(post.orders[c], post.sessions[c],
                                                      post.orders[t], post.sessions[t]).pvalue
    out["ratio: delta method"] = S.ratio_delta(post.orders[c], post.sessions[c],
                                               post.orders[t], post.sessions[t]).pvalue
    if with_bootstrap:
        out["GMV: Poisson bootstrap"] = S.poisson_bootstrap(post.gmv[c], post.gmv[t], 300, rng).pvalue
    out["_rho2"], out["_r2"] = rho2, r2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--boot-runs", type=int, default=150, help="bootstrap is slower: fewer runs")
    args = ap.parse_args()
    rng = np.random.default_rng(2026)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    aa = pd.DataFrame([one_run(rng, 0.0, i < args.boot_runs) for i in range(args.runs)])
    ab = pd.DataFrame([one_run(rng, UPLIFT, i < args.boot_runs) for i in range(args.runs)])

    methods = [m for m in aa.columns if not m.startswith("_")]
    table = pd.DataFrame({
        "method": methods,
        "runs": [aa[m].notna().sum() for m in methods],
        "FPR (A/A), %": [100 * (aa[m].dropna() < 0.05).mean() for m in methods],
        f"power (+{100 * UPLIFT:.0f} % conv.), %": [100 * (ab[m].dropna() < 0.05).mean() for m in methods],
    })
    # 95 % binomial band around FPR = 5 %, outside of it the test is miscalibrated
    half = 1.96 * np.sqrt(0.05 * 0.95 / args.runs) * 100
    rho2, r2 = aa["_rho2"].mean(), aa["_r2"].mean()

    p_t = (ab["GMV: Welch t-test"] < 0.05).mean()
    p_cm = (ab["GMV: CUPED, 3 covariates"] < 0.05).mean()
    lines = [
        "# Monte Carlo validation\n",
        f"{args.runs} A/A + {args.runs} A/B experiments, {N_USERS:,} users each, "
        f"pre-period {PRE_DAYS} d, experiment {EXP_DAYS} d. Bootstrap: first {args.boot_runs} runs.\n",
        table.to_markdown(index=False, floatfmt=".1f"),
        f"\nAcceptable FPR band for {args.runs} runs: 5.0 ± {half:.1f} %.",
        f"\nMean variance reduction: CUPED rho^2 = {rho2:.3f}; 3 covariates R^2 = {r2:.3f} "
        f"-> sample size needed for the same power is {100 * r2:.0f} % smaller.",
        f"Power of GMV test: {100 * p_t:.0f} % -> {100 * p_cm:.0f} % with regression-adjusted CUPED.",
    ]
    (ROOT / "results" / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # p-value histograms under H0, a calibrated test gives a flat histogram
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axes, ["ratio: naive session-level", "ratio: delta method"]):
        ax.hist(aa[m], bins=20, range=(0, 1), color="tab:red" if "naive" in m else "tab:green", alpha=.8)
        ax.axhline(args.runs / 20, ls="--", c="k", lw=1)
        ax.set_title(f"{m}\nFPR = {100 * (aa[m] < 0.05).mean():.1f} %")
        ax.set_xlabel("p-value under A/A")
    axes[0].set_ylabel("count")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "aa_pvalues_ratio.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(table["method"], table.iloc[:, 3], color="tab:blue")
    ax.set_xlabel(f"power at +{100 * UPLIFT:.0f} % conversion, %"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "power.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
