-- Revenue views: by zone, by hour-of-week, and the reconciliation checks.

CREATE OR REPLACE TABLE mart_zone_summary AS
SELECT
    z.zone_id, z.zone_name, z.borough, z.service_zone, z.is_airport,
    SUM(d.trips)        AS trips,
    SUM(d.revenue)      AS revenue,
    SUM(d.driver_hours) AS driver_hours,
    SUM(d.revenue) / NULLIF(SUM(d.driver_hours), 0) AS revenue_per_driver_hour,
    SUM(d.revenue) / NULLIF(SUM(d.trips), 0)        AS mean_fare,
    COUNT(DISTINCT d.hour_of_week) AS active_hours_of_week
FROM mart_demand d
JOIN dim_zone z ON z.zone_id = d.zone_id
GROUP BY 1,2,3,4,5
ORDER BY revenue DESC;

CREATE OR REPLACE TABLE mart_hourly_profile AS
SELECT
    hour_of_week, ANY_VALUE(dow_name) AS dow_name, hour,
    ANY_VALUE(is_weekend) AS is_weekend, ANY_VALUE(daypart) AS daypart,
    SUM(trips)   AS trips,
    SUM(revenue) AS revenue,
    SUM(driver_hours) AS driver_hours,
    SUM(revenue) / NULLIF(SUM(driver_hours), 0) AS revenue_per_driver_hour,
    COUNT(DISTINCT zone_id) AS active_zones
FROM mart_demand
GROUP BY hour_of_week, hour
ORDER BY hour_of_week;

CREATE OR REPLACE TABLE mart_monthly AS
SELECT month,
       SUM(trips) AS trips, SUM(revenue) AS revenue,
       SUM(driver_hours) AS driver_hours,
       SUM(revenue) / NULLIF(SUM(driver_hours), 0) AS revenue_per_driver_hour
FROM mart_demand GROUP BY month ORDER BY month;

-- Reconciliation. src/build.py aborts if any of these disagree: a mart that does not tie
-- back to the partitions is a mart nobody should build an allocation on.
CREATE OR REPLACE TABLE reconciliation AS
SELECT 'demand_trips_equal_fact_trips' AS check_name,
       (SELECT SUM(trips) FROM mart_demand)          AS left_value,
       (SELECT COUNT(*) FROM fact_trip)              AS right_value
UNION ALL
SELECT 'demand_revenue_equals_fact_revenue',
       (SELECT ROUND(SUM(revenue)) FROM mart_demand),
       (SELECT ROUND(SUM(fare_total)) FROM fact_trip)
UNION ALL
SELECT 'profile_trips_equal_demand_trips',
       (SELECT SUM(total_trips) FROM mart_demand_profile),
       (SELECT SUM(trips) FROM mart_demand)
UNION ALL
SELECT 'zone_summary_ties_to_demand',
       (SELECT SUM(trips) FROM mart_zone_summary),
       (SELECT SUM(trips) FROM mart_demand)
UNION ALL
SELECT 'hourly_profile_ties_to_demand',
       (SELECT SUM(trips) FROM mart_hourly_profile),
       (SELECT SUM(trips) FROM mart_demand)
UNION ALL
SELECT 'monthly_ties_to_demand',
       (SELECT SUM(trips) FROM mart_monthly),
       (SELECT SUM(trips) FROM mart_demand)
UNION ALL
SELECT 'hour_of_week_spine_is_complete',
       (SELECT COUNT(*) FROM dim_time), 168
UNION ALL
SELECT 'no_unknown_zones_survived',
       (SELECT COUNT(*) FROM mart_demand WHERE zone_id IN (264, 265)), 0
UNION ALL
SELECT 'every_demand_zone_exists_in_dim',
       (SELECT COUNT(*) FROM mart_demand d
          LEFT JOIN dim_zone z ON z.zone_id = d.zone_id WHERE z.zone_id IS NULL), 0;
