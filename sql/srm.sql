-- Sample Ratio Mismatch check: how many users actually landed in each variant.
SELECT variant, COUNT(*) AS n_users
FROM assignments
WHERE experiment = :experiment
GROUP BY variant
ORDER BY variant;
