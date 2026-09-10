-- Zone dimension, from TLC's own lookup table.
--
-- 265 zones across five boroughs. The lookup is the authoritative mapping from the
-- LocationID codes in the trip records to names and boroughs, and joining through it is
-- what turns "zone 132" into "JFK Airport" in every output a human reads.
--
-- The service_zone column matters for the allocation: Airports, Yellow Zone (core
-- Manhattan), Boro Zone and EWR behave completely differently, and an allocation that
-- treated a 3am airport run and a 3am outer-borough run as the same opportunity would
-- be wrong in a way that averages hide.

CREATE OR REPLACE TABLE dim_zone AS
SELECT
    CAST(LocationID AS INTEGER) AS zone_id,
    Borough                     AS borough,
    Zone                        AS zone_name,
    service_zone,
    -- Airports are the highest-value cells in the whole dataset and the allocation
    -- finds them; flagging them explicitly makes that checkable rather than incidental.
    CASE WHEN service_zone = 'Airports' THEN true ELSE false END AS is_airport
FROM read_csv(getvariable('zone_path'), header = true, auto_detect = true)
WHERE CAST(LocationID AS INTEGER) NOT IN (264, 265);   -- Unknown / NA are not places

CREATE OR REPLACE TABLE dim_time AS
-- The hour-of-week spine: 7 days x 24 hours = 168 cells. Generated rather than derived
-- from the data, so a cell with genuinely zero demand still exists as a row. Deriving it
-- from observed trips would silently drop the quiet cells, and "nobody wants a cab here
-- at 4am" is exactly the information an allocation needs.
WITH dows AS (SELECT * FROM (VALUES (0,'Sunday'),(1,'Monday'),(2,'Tuesday'),
                                    (3,'Wednesday'),(4,'Thursday'),(5,'Friday'),
                                    (6,'Saturday')) AS t(dow, dow_name)),
     hours AS (SELECT UNNEST(range(0, 24)) AS hour)
SELECT
    d.dow,
    d.dow_name,
    h.hour,
    d.dow * 24 + h.hour AS hour_of_week,
    CASE WHEN d.dow IN (0, 6) THEN true ELSE false END AS is_weekend,
    CASE
        WHEN h.hour BETWEEN 6 AND 9   THEN 'morning_peak'
        WHEN h.hour BETWEEN 10 AND 15 THEN 'midday'
        WHEN h.hour BETWEEN 16 AND 19 THEN 'evening_peak'
        WHEN h.hour BETWEEN 20 AND 23 THEN 'evening'
        ELSE 'overnight'
    END AS daypart
FROM dows d CROSS JOIN hours h
ORDER BY hour_of_week;
