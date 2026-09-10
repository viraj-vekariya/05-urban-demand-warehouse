-- The trip fact, read straight from the partitioned Parquet.
--
-- A VIEW, not a table. The partitions are already columnar, compressed and pruned by
-- month; materialising 39 million rows into DuckDB again would double the storage and
-- buy nothing, because every downstream query aggregates rather than scanning rows.
--
-- The Hive-style path (month=YYYY-MM/) lets DuckDB prune partitions from the filename,
-- so the backtest's "train on months 1-9" reads nine files rather than twelve.

CREATE OR REPLACE VIEW fact_trip AS
SELECT
    pickup_ts, dropoff_ts,
    pickup_zone_id, dropoff_zone_id,
    passengers, distance_mi, fare_total, fare_base, tip,
    duration_sec,
    duration_sec / 3600.0 AS duration_hours,
    pickup_hour, pickup_date,
    -- DuckDB's DAYOFWEEK is 0=Sunday, matching dim_time.
    pickup_dow AS dow,
    pickup_dow * 24 + pickup_hour AS hour_of_week,
    month
FROM read_parquet(getvariable('warehouse_glob'), hive_partitioning = false);
