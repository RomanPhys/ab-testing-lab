"""
Builds an event-level SQLite warehouse (users, assignments, sessions, orders) from the generative
model, so the analysis goes through SQL like it would on a real platform.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .population import VALUE_SIGMA, Population, hash_split

SCHEMA = """
CREATE TABLE users       (user_id INTEGER PRIMARY KEY, platform TEXT, region TEXT);
CREATE TABLE assignments (experiment TEXT, user_id INTEGER, variant TEXT,
                          PRIMARY KEY (experiment, user_id));
CREATE TABLE sessions    (session_id INTEGER PRIMARY KEY, user_id INTEGER, day INTEGER);
CREATE TABLE orders      (order_id INTEGER PRIMARY KEY, session_id INTEGER, user_id INTEGER,
                          day INTEGER, gmv REAL);
CREATE INDEX ix_sessions_user ON sessions(user_id);
CREATE INDEX ix_orders_user   ON orders(user_id);
"""


def build_warehouse(path: str | Path, pop: Population, experiment: str, pre_days: int,
                    exp_days: int, uplift: float, rng: np.random.Generator) -> np.ndarray:
    """Writes the database and returns the treatment flags (used only in checks)."""
    path = Path(path)
    path.unlink(missing_ok=True)
    user_ids = np.arange(pop.n)
    treated = hash_split(user_ids, salt=experiment)

    # sessions: one row per session, day in [-pre_days, exp_days)
    n_sess = rng.poisson(pop.lam * (pre_days + exp_days))
    s_user = np.repeat(user_ids, n_sess)
    s_day = rng.integers(-pre_days, exp_days, size=s_user.size)

    # orders: Bernoulli per session, the treatment works only during the experiment
    p = pop.p[s_user] * (1.0 + uplift * (treated[s_user] == 1) * (s_day >= 0))
    has_order = rng.random(s_user.size) < np.clip(p, 0, 1)
    o_sess = np.flatnonzero(has_order)
    o_user = s_user[o_sess]
    o_gmv = np.round(rng.lognormal(pop.m[o_user], VALUE_SIGMA), 2)

    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO users VALUES (?,?,?)",
                    zip(user_ids.tolist(), pop.platform.tolist(), pop.region.tolist()))
    con.executemany("INSERT INTO assignments VALUES (?,?,?)",
                    ((experiment, int(u), "treatment" if t else "control") for u, t in zip(user_ids, treated)))
    con.executemany("INSERT INTO sessions VALUES (?,?,?)",
                    zip(range(s_user.size), s_user.tolist(), s_day.tolist()))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)",
                    zip(range(o_sess.size), o_sess.tolist(), o_user.tolist(),
                        s_day[o_sess].tolist(), o_gmv.tolist()))
    con.commit()
    con.close()
    return treated


def run_sql(path: str | Path, sql_file: str | Path, **params) -> pd.DataFrame:
    with sqlite3.connect(path) as con:
        return pd.read_sql_query(Path(sql_file).read_text(encoding="utf-8"), con, params=params)
