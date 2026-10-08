-- Витрина для BI: один ряд = (день × вариант × платформа).
-- Денормализована намеренно: BI-инструмент (DataLens, Superset, Metabase) агрегирует
-- суммы и считает отношения сам, поэтому в витрину кладутся АДДИТИВНЫЕ величины
-- (gmv, orders, sessions, users), а не готовые средние: среднее от средних по дням
-- не равно общему среднему, а сумма сумм — равна.
WITH u AS (
    SELECT a.user_id, a.variant, us.platform, us.region
    FROM assignments AS a
    JOIN users AS us USING (user_id)
    WHERE a.experiment = :experiment
),
-- активные пользователи считаются отдельно: COUNT(DISTINCT) по дню нельзя
-- восстановить из суточных сумм, поэтому эта метрика кладётся в витрину готовой
sess AS (
    SELECT u.variant, u.platform, s.day,
           COUNT(*)                     AS sessions,
           COUNT(DISTINCT s.user_id)    AS active_users
    FROM sessions AS s
    JOIN u USING (user_id)
    WHERE s.day >= 0
    GROUP BY u.variant, u.platform, s.day
),
ord AS (
    SELECT u.variant, u.platform, o.day,
           COUNT(*)      AS orders,
           SUM(o.gmv)    AS gmv,
           COUNT(DISTINCT o.user_id) AS buyers
    FROM orders AS o
    JOIN u USING (user_id)
    WHERE o.day >= 0
    GROUP BY u.variant, u.platform, o.day
),
-- размер группы фиксирован на старте эксперимента: делить дневной GMV надо
-- на всех назначенных, а не на активных, иначе ITT превращается в per-protocol
sizes AS (
    SELECT variant, platform, COUNT(*) AS assigned_users
    FROM u GROUP BY variant, platform
)
SELECT s.day,
       s.variant,
       s.platform,
       z.assigned_users,
       s.active_users,
       s.sessions,
       COALESCE(o.orders, 0)   AS orders,
       COALESCE(o.buyers, 0)   AS buyers,
       COALESCE(o.gmv,    0.0) AS gmv,
       SUM(COALESCE(o.gmv, 0.0)) OVER (
           PARTITION BY s.variant, s.platform ORDER BY s.day
       )                        AS gmv_cum
FROM sess AS s
LEFT JOIN ord AS o ON o.variant = s.variant AND o.platform = s.platform AND o.day = s.day
JOIN sizes AS z   ON z.variant = s.variant AND z.platform = s.platform
ORDER BY s.day, s.variant, s.platform;
