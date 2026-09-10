-- THE DEMAND WAREHOUSE: zone x hour x weekday.
--
-- This is the analytical spine. Everything downstream - the forecast, the allocation, the
-- backtest - reads this table and nothing else, so it is worth being precise about what a
-- "cell" means and how its metrics are defined.
--
-- A CELL is (pickup zone, day of week, hour of day). 265 zones x 168 hour-of-week slots
-- is ~44,500 possible cells; most are genuinely empty and that is information.
--
-- THE KEY METRIC: revenue per driver-hour. Not revenue, and not trip count.
--   * total revenue rewards cells that are merely busy;
--   * trip count rewards short cheap trips;
--   * revenue per driver-hour is what a driver actually optimises - it accounts for the
--     fact that a $70 airport run occupying 50 minutes beats four $12 crosstown hops that
--     occupy the same 50 minutes plus the deadheading between them.
-- Getting this wrong is the single most common error in this genre of analysis: an
-- allocation optimising trip count sends the whole fleet to Midtown at lunchtime.

CREATE OR REPLACE TABLE mart_demand AS
WITH observed AS (
    SELECT
        pickup_zone_id AS zone_id,
        dow,
        pickup_hour    AS hour,
        month,
        COUNT(*)                       AS trips,
        SUM(fare_total)                AS revenue,
        SUM(duration_hours)            AS driver_hours,
        AVG(fare_total)                AS mean_fare,
        AVG(distance_mi)               AS mean_distance,
        AVG(duration_sec) / 60.0       AS mean_duration_min,
        SUM(tip)                       AS tips,
        COUNT(DISTINCT pickup_date)    AS days_observed
    FROM fact_trip
    GROUP BY 1, 2, 3, 4
)
SELECT
    o.zone_id,
    z.zone_name,
    z.borough,
    z.service_zone,
    z.is_airport,
    o.dow,
    t.dow_name,
    o.hour,
    t.hour_of_week,
    t.is_weekend,
    t.daypart,
    o.month,
    o.trips,
    o.revenue,
    o.driver_hours,
    o.mean_fare,
    o.mean_distance,
    o.mean_duration_min,
    o.tips,
    o.days_observed,
    -- Per-day rates, so cells observed over different numbers of days are comparable.
    -- A month has 4 or 5 of each weekday; comparing raw totals across months would
    -- rank a 5-Monday month above a 4-Monday one for no real reason.
    o.trips / NULLIF(o.days_observed, 0)   AS trips_per_day,
    o.revenue / NULLIF(o.days_observed, 0) AS revenue_per_day,
    -- THE ALLOCATION METRIC.
    o.revenue / NULLIF(o.driver_hours, 0)  AS revenue_per_driver_hour
FROM observed o
JOIN dim_zone z ON z.zone_id = o.zone_id
JOIN dim_time t ON t.dow = o.dow AND t.hour = o.hour;

-- The month-agnostic profile: average behaviour of each cell across all training months.
-- This is what the allocation is built from, because tomorrow is not a month we have.
CREATE OR REPLACE TABLE mart_demand_profile AS
SELECT
    zone_id,
    ANY_VALUE(zone_name)     AS zone_name,
    ANY_VALUE(borough)       AS borough,
    ANY_VALUE(service_zone)  AS service_zone,
    ANY_VALUE(is_airport)    AS is_airport,
    dow,
    hour,
    ANY_VALUE(hour_of_week)  AS hour_of_week,
    ANY_VALUE(daypart)       AS daypart,
    COUNT(DISTINCT month)    AS months_observed,
    SUM(trips)               AS total_trips,
    SUM(revenue)             AS total_revenue,
    SUM(driver_hours)        AS total_driver_hours,
    AVG(trips_per_day)       AS mean_trips_per_day,
    AVG(revenue_per_day)     AS mean_revenue_per_day,
    -- Computed from the SUMS, not as an average of ratios. Averaging per-month ratios
    -- weights a month with 3 trips the same as one with 30,000 - a classic and very
    -- easy mistake that makes tiny cells look enormously profitable.
    SUM(revenue) / NULLIF(SUM(driver_hours), 0) AS revenue_per_driver_hour,
    STDDEV_SAMP(trips_per_day) AS trips_per_day_sd
FROM mart_demand
GROUP BY zone_id, dow, hour;
