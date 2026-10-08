-- Per-user metrics for one experiment: experiment period (day >= 0) and pre-period (day < 0).
-- Randomisation unit = user, so every row is one independent observation.
-- LEFT JOINs keep users who were assigned but did nothing: dropping them would bias the comparison
-- (intent-to-treat).
WITH exp_users AS (
    SELECT a.user_id, a.variant, u.platform, u.region
    FROM assignments AS a
    JOIN users AS u USING (user_id)
    WHERE a.experiment = :experiment
),
s AS (
    SELECT user_id,
           SUM(CASE WHEN day >= 0 THEN 1 ELSE 0 END) AS sessions,
           SUM(CASE WHEN day <  0 THEN 1 ELSE 0 END) AS sessions_pre
    FROM sessions
    GROUP BY user_id
),
o AS (
    SELECT user_id,
           SUM(CASE WHEN day >= 0 THEN 1   ELSE 0 END) AS orders,
           SUM(CASE WHEN day >= 0 THEN gmv ELSE 0 END) AS gmv,
           SUM(CASE WHEN day <  0 THEN 1   ELSE 0 END) AS orders_pre,
           SUM(CASE WHEN day <  0 THEN gmv ELSE 0 END) AS gmv_pre,
           -- last 14 pre-period days: a "historical experiment" of the same length, for power analysis
           SUM(CASE WHEN day >= -14 AND day < 0 THEN gmv ELSE 0 END) AS gmv_hist14
    FROM orders
    GROUP BY user_id
)
SELECT e.user_id,
       e.variant,
       e.platform,
       e.region,
       COALESCE(s.sessions, 0)     AS sessions,
       COALESCE(s.sessions_pre, 0) AS sessions_pre,
       COALESCE(o.orders, 0)       AS orders,
       COALESCE(o.orders_pre, 0)   AS orders_pre,
       COALESCE(o.gmv, 0.0)        AS gmv,
       COALESCE(o.gmv_pre, 0.0)    AS gmv_pre,
       COALESCE(o.gmv_hist14, 0.0) AS gmv_hist14
FROM exp_users AS e
LEFT JOIN s USING (user_id)
LEFT JOIN o USING (user_id);
