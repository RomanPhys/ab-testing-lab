"""
One experiment from start to finish: SQLite warehouse, SQL, SRM check, metrics with CI, CUPED,
segments and the decision.

    python scripts/run_case_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ablab.db import build_warehouse, run_sql  # noqa: E402
from ablab.population import sample_population  # noqa: E402
from ablab import stats as S  # noqa: E402

SEED, N_USERS, PRE_DAYS, EXP_DAYS = 7, 60_000, 28, 14
EXPERIMENT, TRUE_UPLIFT = "new_checkout_button", 0.08   # +8 % to P(order | session)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, controls FWER across segments."""
    m, order = len(pvals), np.argsort(pvals)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def fmt(res: S.TestResult, base: float) -> dict:
    return {"effect_abs": res.effect, "effect_rel_%": 100 * res.effect / base,
            "ci95_rel_%": f"[{100 * res.ci_low / base:+.2f}, {100 * res.ci_high / base:+.2f}]",
            "p_value": res.pvalue}


def main() -> None:
    rng = np.random.default_rng(SEED)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    db = ROOT / "data" / "warehouse.sqlite"
    db.parent.mkdir(exist_ok=True)

    pop = sample_population(N_USERS, rng)
    build_warehouse(db, pop, EXPERIMENT, PRE_DAYS, EXP_DAYS, TRUE_UPLIFT, rng)

    srm = run_sql(db, ROOT / "sql" / "srm.sql", experiment=EXPERIMENT)
    n_c, n_t = srm.set_index("variant").loc[["control", "treatment"], "n_users"]
    p_srm = S.srm_test(int(n_c), int(n_t))

    df = run_sql(db, ROOT / "sql" / "user_metrics.sql", experiment=EXPERIMENT)
    # design comes before looking at the outcome: MDE on a past 14-day window of the same users
    hist = df.gmv_hist14.values
    mde_rel = S.mde(hist.std(ddof=1), len(df) // 2) / hist.mean()

    c, t = df[df.variant == "control"], df[df.variant == "treatment"]
    base_gmv, base_conv = c.gmv.mean(), (c.orders > 0).mean()
    base_ratio = c.orders.sum() / c.sessions.sum()

    rows = []
    r = S.welch_ttest(c.gmv.values, t.gmv.values)
    rows.append({"metric": "GMV per user", "method": "Welch t-test", **fmt(r, base_gmv)})
    r_cup, rho2 = S.cuped(c.gmv.values, t.gmv.values, c.gmv_pre.values, t.gmv_pre.values)
    rows.append({"metric": "GMV per user", "method": "CUPED (covariate: pre-period GMV)", **fmt(r_cup, base_gmv)})
    cov = ["gmv_pre", "orders_pre", "sessions_pre"]
    r_multi, r2 = S.cuped_multi(c.gmv.values, t.gmv.values, c[cov].values, t[cov].values)
    rows.append({"metric": "GMV per user", "method": "CUPED, 3 covariates (regression adj.)", **fmt(r_multi, base_gmv)})
    r = S.poisson_bootstrap(c.gmv.values, t.gmv.values, n_boot=1000, rng=rng)
    rows.append({"metric": "GMV per user", "method": "Poisson bootstrap", **fmt(r, base_gmv)})
    r = S.welch_ttest((c.orders > 0).values.astype(float), (t.orders > 0).values.astype(float))
    rows.append({"metric": "Buyer conversion", "method": "Welch t-test", **fmt(r, base_conv)})
    r = S.ratio_delta(c.orders.values, c.sessions.values, t.orders.values, t.sessions.values)
    rows.append({"metric": "Orders per session", "method": "Delta method", **fmt(r, base_ratio)})
    r = S.ratio_naive(c.orders.values, c.sessions.values, t.orders.values, t.sessions.values)
    rows.append({"metric": "Orders per session", "method": "Naive session-level z-test (wrong)", **fmt(r, base_ratio)})
    res = pd.DataFrame(rows)

    # segments: does the effect differ between android, ios and web? several tests, so Holm correction
    seg = []
    for plat, g in df.groupby("platform"):
        gc, gt = g[g.variant == "control"], g[g.variant == "treatment"]
        rr, _ = S.cuped_multi(gc.gmv.values, gt.gmv.values, gc[cov].values, gt[cov].values)
        seg.append({"platform": plat, "n_users": len(g), "effect_rel_%": 100 * rr.effect / gc.gmv.mean(),
                    "p_value": rr.pvalue})
    seg = pd.DataFrame(seg)
    seg["p_holm"] = holm(seg.p_value.tolist())

    daily = run_sql(db, ROOT / "sql" / "daily_gmv.sql", experiment=EXPERIMENT)
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, g in daily.groupby("variant"):
        ax.plot(g.day, g.cum_gmv_per_user, marker="o", ms=3, label=v)
    ax.set_xlabel("experiment day"); ax.set_ylabel("cumulative GMV per user, RUB")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / "figures" / "case_cumulative_gmv.png", dpi=150); plt.close(fig)

    se_ratio = res.loc[res.method.str.startswith("Delta")].iloc[0]
    lines = [
        f"# Case study: `{EXPERIMENT}`\n",
        f"Users: {len(df):,}; pre-period {PRE_DAYS} d, experiment {EXP_DAYS} d. "
        f"True effect injected: +{100 * TRUE_UPLIFT:.0f} % to P(order | session).\n",
        "## 0. Design\n",
        f"MDE for GMV per user (alpha = 0.05, power = 0.8, {len(df) // 2:,} users per group), "
        f"estimated on a historical 14-day window: **{100 * mde_rel:.1f} %** with a plain t-test; "
        f"variance reduction lowers it by a factor sqrt(1 - R^2) (see below).\n",
        "## 1. Sample Ratio Mismatch\n",
        srm.to_markdown(index=False),
        f"\nchi-square p = {p_srm:.3f} -> " + ("split is healthy" if p_srm > 1e-3 else "**SRM, results invalid**"),
        "\n## 2. Metrics\n",
        res.to_markdown(index=False, floatfmt=".4g"),
        f"\nVariance reduction: CUPED rho^2 = {rho2:.3f}, 3 covariates R^2 = {r2:.3f} -> "
        f"the same power with {100 * r2:.0f} % fewer users.",
        "\n## 3. Segments (CUPED, Holm-corrected)\n",
        seg.to_markdown(index=False, floatfmt=".4g"),
        "\n## 4. Decision\n",
        "Decide on the pre-registered primary metric (GMV per user, CUPED with 3 covariates). Segment differences are "
        "exploratory unless they survive the correction.",
    ]
    (ROOT / "results" / "case_study.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
