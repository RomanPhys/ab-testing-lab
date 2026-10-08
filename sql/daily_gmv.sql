-- Daily GMV per user by variant (window function: cumulative GMV for the "peeking" plot).
WITH daily AS (
    SELECT a.variant, o.day, SUM(o.gmv) AS gmv
    FROM orders AS o
    JOIN assignments AS a ON a.user_id = o.user_id AND a.experiment = :experiment
    WHERE o.day >= 0
    GROUP BY a.variant, o.day
),
sizes AS (
    SELECT variant, COUNT(*) AS n_users FROM assignments WHERE experiment = :experiment GROUP BY variant
)
SELECT d.variant,
       d.day,
       d.gmv / s.n_users                                                  AS gmv_per_user,
       SUM(d.gmv) OVER (PARTITION BY d.variant ORDER BY d.day) / s.n_users AS cum_gmv_per_user
FROM daily AS d
JOIN sizes AS s USING (variant)
ORDER BY d.variant, d.day;
