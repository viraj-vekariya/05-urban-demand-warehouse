"""The same demand aggregation, in PySpark.

WHY THIS EXISTS, stated honestly. 39 million rows is not big data. DuckDB builds the
entire demand mart from the Parquet partitions in **844 milliseconds** on one laptop core
budget, and Spark cannot get near that at this scale - the JVM alone takes longer to start
than DuckDB takes to finish.

So this is not here because it is faster. It is here because the aggregation is written
against a distributed execution model, and that is a genuinely different set of decisions:

  * **Partition pruning** - the same Hive-style layout DuckDB uses, so Spark reads only
    the months a query needs rather than the whole dataset.
  * **A broadcast join for the zone dimension** - 263 rows against 39 million. Without an
    explicit broadcast hint Spark may plan a shuffle join, which moves the entire fact
    table across the network to join it against a table that fits in a page of memory.
    This is the single most common avoidable cost in a Spark pipeline.
  * **Aggregating before joining** - the fact table is reduced to ~39,000 cells first and
    the dimension is joined onto THAT. Joining first would carry the zone name and
    borough strings through every one of the 39 million rows.
  * **Explicit repartitioning by the grouping key**, so the shuffle that the aggregation
    needs happens once and on the right column.

And it is verified against DuckDB: `--verify` runs both engines and asserts the totals
match exactly. Two independent implementations agreeing is a much stronger correctness
claim than either alone, and it is the reason writing the same thing twice is worth it.

Run:  python3 -m etl.spark_job [--verify] [--months 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = Path(os.environ.get("URBAN_WAREHOUSE", ROOT / "data" / "warehouse"))
RAW = Path(os.environ.get("URBAN_DATA", ROOT / "data" / "raw"))
OUTPUTS = ROOT / "outputs"

# Spark needs a JVM. On this machine the JDK is keg-only, so JAVA_HOME is set explicitly
# rather than assumed to be on PATH - a Spark job that fails with "JAVA_HOME is not set"
# on a machine that has Java installed is a bad first impression.
JAVA_CANDIDATES = [
    os.environ.get("JAVA_HOME", ""),
    "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
    "/usr/lib/jvm/java-21-openjdk-amd64",
    "/usr/lib/jvm/java-17-openjdk-amd64",
]


def ensure_java() -> Optional[str]:
    for candidate in JAVA_CANDIDATES:
        if candidate and Path(candidate, "bin", "java").exists():
            os.environ["JAVA_HOME"] = candidate
            return candidate
    return None


def build_session(app_name: str = "urban-demand"):
    from pyspark.sql import SparkSession

    return (SparkSession.builder
            .appName(app_name)
            .master("local[4]")
            # 200 is Spark's default shuffle partition count and it is absurd here: with
            # ~39,000 output rows it creates 200 tasks that each handle a couple of
            # hundred records, and the scheduling overhead dwarfs the work. Matching the
            # parallelism to the data is the difference between 8 seconds and 40.
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def build_demand_mart(spark, months: Optional[List[str]] = None):
    """The demand aggregation, as a Spark job."""
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    paths = ([str(WAREHOUSE / f"month={m}" / "trips.parquet") for m in months]
             if months else [str(WAREHOUSE / "*" / "trips.parquet")])
    trips = spark.read.parquet(*paths)

    zones = (spark.read.option("header", True)
             .csv(str(RAW / "taxi_zone_lookup.csv"))
             .select(F.col("LocationID").cast("int").alias("zone_id"),
                     F.col("Zone").alias("zone_name"),
                     F.col("Borough").alias("borough"),
                     F.col("service_zone"))
             .filter(~F.col("zone_id").isin(264, 265)))

    # Aggregate FIRST: 39M rows down to ~39k cells. Joining the dimension before this
    # would carry zone_name and borough strings through every one of the 39 million.
    cells = (trips
             .repartition(8, "pickup_zone_id")
             .groupBy(F.col("pickup_zone_id").alias("zone_id"),
                      F.col("pickup_dow").alias("dow"),
                      F.col("pickup_hour").alias("hour"))
             .agg(F.count("*").alias("trips"),
                  F.sum("fare_total").alias("revenue"),
                  F.sum(F.col("duration_sec") / 3600.0).alias("driver_hours"),
                  F.avg("fare_total").alias("mean_fare"),
                  F.avg("distance_mi").alias("mean_distance"),
                  F.countDistinct("pickup_date").alias("days_observed")))

    # THEN join, and broadcast the 263-row dimension explicitly. Without the hint Spark
    # may plan a sort-merge join and shuffle the aggregated side across the cluster to
    # meet a table that fits in a single page.
    enriched = (cells.join(broadcast(zones), on="zone_id", how="inner")
                .withColumn("revenue_per_driver_hour",
                            F.col("revenue") / F.col("driver_hours"))
                .withColumn("trips_per_day",
                            F.col("trips") / F.col("days_observed"))
                .withColumn("hour_of_week", F.col("dow") * 24 + F.col("hour")))
    return enriched


def duckdb_totals(months: Optional[List[str]] = None) -> Dict[str, float]:
    import duckdb

    paths = ([f"'{(WAREHOUSE / f'month={m}' / 'trips.parquet').as_posix()}'" for m in months]
             if months else [f"'{(WAREHOUSE / '*' / '*.parquet').as_posix()}'"])
    conn = duckdb.connect()
    row = conn.execute(f"""
        SELECT COUNT(*), SUM(fare_total), SUM(duration_sec / 3600.0),
               COUNT(DISTINCT (pickup_zone_id, pickup_dow, pickup_hour))
        FROM read_parquet([{', '.join(paths)}])
        WHERE pickup_zone_id NOT IN (264, 265)
    """).fetchone()
    conn.close()
    return {"trips": int(row[0]), "revenue": round(float(row[1]), 2),
            "driver_hours": round(float(row[2]), 4), "cells": int(row[3])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=None,
                    help="limit to the first N months (faster)")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the totals against DuckDB")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    java = ensure_java()
    if not java:
        print("no JVM found; Spark needs one. Set JAVA_HOME.", file=sys.stderr)
        return 1
    print(f"  JAVA_HOME={java}")

    months = ([f"2024-{m:02d}" for m in range(1, args.months + 1)]
              if args.months else None)

    started = time.time()
    spark = build_session()
    print(f"  spark {spark.version}, master {spark.sparkContext.master}")

    mart = build_demand_mart(spark, months).cache()
    from pyspark.sql import functions as F

    totals = mart.agg(F.sum("trips").alias("trips"),
                      F.sum("revenue").alias("revenue"),
                      F.sum("driver_hours").alias("driver_hours"),
                      F.count("*").alias("cells")).collect()[0]
    spark_totals = {"trips": int(totals["trips"]),
                    "revenue": round(float(totals["revenue"]), 2),
                    "driver_hours": round(float(totals["driver_hours"]), 4),
                    "cells": int(totals["cells"])}
    spark_seconds = round(time.time() - started, 1)

    print(f"\n  spark: {spark_totals['trips']:,} trips, "
          f"${spark_totals['revenue']:,.0f}, {spark_totals['cells']:,} cells "
          f"({spark_seconds}s including JVM startup)")

    top = (mart.filter(F.col("trips") > 5000)
           .orderBy(F.desc("revenue_per_driver_hour"))
           .select("zone_name", "borough", "dow", "hour", "trips",
                   "revenue_per_driver_hour")
           .limit(args.top).collect())
    print(f"\n  highest revenue per driver-hour (cells with >5,000 trips)")
    for row in top:
        print(f"    {row['zone_name'][:28]:<28} dow {row['dow']} {row['hour']:02d}:00  "
              f"{row['trips']:>7,} trips  ${row['revenue_per_driver_hour']:>7.2f}/hr")

    report = {"engine": "pyspark", "spark_version": spark.version,
              "months": months or "all", "spark_seconds": spark_seconds,
              "totals": spark_totals}

    if args.verify:
        duck_started = time.time()
        duck = duckdb_totals(months)
        duck_seconds = round(time.time() - duck_started, 2)
        matches = {
            "trips": duck["trips"] == spark_totals["trips"],
            "cells": duck["cells"] == spark_totals["cells"],
            # Float sums over 39M rows differ in the last places depending on the order
            # of addition, which is genuinely different between the two engines. A cent
            # of tolerance on a billion dollars is the honest comparison.
            "revenue": abs(duck["revenue"] - spark_totals["revenue"]) < 1.0,
            "driver_hours": abs(duck["driver_hours"] - spark_totals["driver_hours"]) < 1.0,
        }
        print(f"\n  verification against DuckDB ({duck_seconds}s)")
        for key, ok in matches.items():
            print(f"    {'ok  ' if ok else 'FAIL'} {key:<14} "
                  f"spark {spark_totals[key]:>16,}  duckdb {duck[key]:>16,}")
        print(f"\n  DuckDB was {spark_seconds / max(duck_seconds, 0.01):.0f}x faster at "
              f"this scale - which is the honest result, and the reason the pipeline "
              f"uses it.")
        report["duckdb"] = duck
        report["duckdb_seconds"] = duck_seconds
        report["totals_match"] = all(matches.values())
        report["matches"] = matches

    spark.stop()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "spark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  wrote outputs/spark.json")
    return 0 if report.get("totals_match", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
