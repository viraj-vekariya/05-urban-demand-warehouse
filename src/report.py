"""Run the forecast and seasonal analysis, and consolidate every measured number.

Run:  python3 -m src.report
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forecast import evaluate                       # noqa: E402
from src.seasonal import decompose, hour_of_week_profile  # noqa: E402

DB = ROOT / "data" / "urban.duckdb"
OUTPUTS = ROOT / "outputs"
# Derived at runtime from the months actually loaded - see src/backtest.split_months for
# why a hard-coded split broke on a reduced dataset.
TRAIN_MONTHS: List[str] = []
TEST_MONTHS: List[str] = []


def forecast_rows(conn, months: List[str]) -> List[Dict[str, object]]:
    """One row per (zone, dow, hour) with the target and the history features.

    History features are computed from the TRAINING months only, and passed to both
    splits. Computing them over all months would leak the test period into the training
    features - the single most common leak in time-series feature engineering, and it
    inflates the model while leaving the baseline honest, which is exactly backwards.
    """
    placeholders = ", ".join(f"'{m}'" for m in months)
    train_placeholders = ", ".join(f"'{m}'" for m in TRAIN_MONTHS)
    rows = conn.execute(f"""
        WITH history AS (
            SELECT zone_id, dow, hour,
                   AVG(trips_per_day)         AS cell_history_mean,
                   COALESCE(STDDEV_SAMP(trips_per_day), 0) AS cell_history_sd
            FROM mart_demand WHERE month IN ({train_placeholders})
            GROUP BY zone_id, dow, hour
        ), zone_history AS (
            SELECT zone_id,
                   AVG(trips_per_day) AS zone_mean_trips,
                   SUM(revenue) / NULLIF(SUM(driver_hours), 0) AS zone_mean_rph
            FROM mart_demand WHERE month IN ({train_placeholders})
            GROUP BY zone_id
        )
        SELECT d.zone_id, d.dow, d.hour,
               AVG(d.trips_per_day)                     AS target,
               ANY_VALUE(d.is_airport)                  AS is_airport,
               COALESCE(ANY_VALUE(h.cell_history_mean), 0) AS cell_history_mean,
               COALESCE(ANY_VALUE(h.cell_history_sd), 0)   AS cell_history_sd,
               COALESCE(ANY_VALUE(z.zone_mean_trips), 0)   AS zone_mean_trips,
               COALESCE(ANY_VALUE(z.zone_mean_rph), 0)     AS zone_mean_revenue_per_hour
        FROM mart_demand d
        LEFT JOIN history h ON h.zone_id = d.zone_id AND h.dow = d.dow AND h.hour = d.hour
        LEFT JOIN zone_history z ON z.zone_id = d.zone_id
        WHERE d.month IN ({placeholders})
        GROUP BY d.zone_id, d.dow, d.hour
    """).fetchall()
    columns = ["zone_id", "dow", "hour", "target", "is_airport", "cell_history_mean",
               "cell_history_sd", "zone_mean_trips", "zone_mean_revenue_per_hour"]
    return [dict(zip(columns, [float(v) if v is not None else 0.0 for v in row]))
            for row in rows]


def main() -> int:
    if not DB.exists():
        print("no warehouse; run python3 -m src.build", file=sys.stderr)
        return 1
    conn = duckdb.connect(str(DB), read_only=True)
    started = time.time()

    global TRAIN_MONTHS, TEST_MONTHS
    from src.backtest import split_months
    TRAIN_MONTHS, TEST_MONTHS = split_months(conn)
    print(f"months: train {TRAIN_MONTHS[0]}..{TRAIN_MONTHS[-1]}, "
          f"test {TEST_MONTHS[0]}..{TEST_MONTHS[-1]}")

    print("forecast: gradient-boosted trees vs the historical-cell-mean baseline")
    train = forecast_rows(conn, TRAIN_MONTHS)
    test = forecast_rows(conn, TEST_MONTHS)
    forecast = evaluate(train, test)
    b, m = forecast["baseline"], forecast["model"]
    print(f"  baseline  MAE {b['mae']:>7.4f}  RMSE {b['rmse']:>7.4f}  R2 {b['r2']:>7.4f}")
    print(f"  model     MAE {m['mae']:>7.4f}  RMSE {m['rmse']:>7.4f}  R2 {m['r2']:>7.4f}")
    print(f"  -> {forecast['verdict']}")
    if m["improvement_over_baseline_pct"] is not None:
        print(f"  -> MAE improvement over the baseline: "
              f"{m['improvement_over_baseline_pct']:+.2f}%")

    print("\nseasonal decomposition of citywide hourly demand")
    hourly = conn.execute("""
        SELECT hour_of_week, SUM(trips) AS trips
        FROM mart_demand GROUP BY hour_of_week ORDER BY hour_of_week""").fetchall()
    # Repeat the 168-hour profile so the decomposition has the two full periods it
    # requires. The profile IS the weekly cycle, so this measures the cycle's shape and
    # strength rather than inventing data.
    series = [float(t) for _, t in hourly] * 3
    profile = hour_of_week_profile(series)
    print(f"  seasonal strength {profile['seasonal_strength']:.4f}")
    print(f"  peak    {profile['peak_label']}  effect {profile['peak_effect']:+,.0f} trips")
    print(f"  trough  {profile['trough_label']}  effect {profile['trough_effect']:+,.0f} trips")
    print(f"  peak-to-trough swing {profile['peak_to_trough']:,.0f} trips")

    zone_top = conn.execute("""
        SELECT zone_name, borough, trips, revenue, revenue_per_driver_hour
        FROM mart_zone_summary WHERE trips > 20000
        ORDER BY revenue_per_driver_hour DESC LIMIT 12""").fetchall()
    zone_volume = conn.execute("""
        SELECT zone_name, borough, trips, revenue, revenue_per_driver_hour
        FROM mart_zone_summary ORDER BY trips DESC LIMIT 12""").fetchall()

    def zone_rows(rows):
        return [{"zone": z, "borough": b, "trips": int(t), "revenue": float(r),
                 "revenue_per_driver_hour": float(rph)} for z, b, t, r, rph in rows]

    # The contrast that motivates the whole allocation metric.
    by_volume = zone_rows(zone_volume)
    by_value = zone_rows(zone_top)
    volume_names = {z["zone"] for z in by_volume}
    value_names = {z["zone"] for z in by_value}

    print("\n  ranking by TRIP VOLUME vs by REVENUE PER DRIVER-HOUR")
    print(f"    top-12 by volume and top-12 by value share only "
          f"{len(volume_names & value_names)} zones")
    print(f"    busiest zone:     {by_volume[0]['zone'][:32]:<32} "
          f"{by_volume[0]['trips']:>9,} trips  ${by_volume[0]['revenue_per_driver_hour']:>7.2f}/hr")
    print(f"    most valuable:    {by_value[0]['zone'][:32]:<32} "
          f"{by_value[0]['trips']:>9,} trips  ${by_value[0]['revenue_per_driver_hour']:>7.2f}/hr")

    backtest = json.loads((OUTPUTS / "backtest.json").read_text()) \
        if (OUTPUTS / "backtest.json").exists() else None
    warehouse = json.loads((OUTPUTS / "warehouse.json").read_text())
    etl = json.loads((OUTPUTS / "etl.json").read_text()) \
        if (OUTPUTS / "etl.json").exists() else None

    tests = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no",
                            "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True)
    test_line = (tests.stdout.strip().splitlines() or ["not run"])[-1]

    report = {
        "project": "Urban Demand Warehouse",
        "role": "CV2 / Data Science + Business Analytics - SQL and data-engineering backbone",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {
            "source": "NYC TLC yellow-cab trip records, 2024, public",
            # Derived, not asserted. This said 12 while the deployed image loads 3, so
            # the API reported a year of data over a quarter of it - every other field
            # here comes from the run, and this one did not.
            "months": etl["months"] if etl else None,
            "source_rows": etl["source_rows"] if etl else None,
            "loaded_rows": warehouse["totals"]["trips"],
            "excluded_pct": etl["excluded_pct"] if etl else None,
            "revenue": warehouse["totals"]["revenue"],
            "driver_hours": warehouse["totals"]["driver_hours"],
            "cells": warehouse["totals"]["cells"],
        },
        "reconciliation": {"checks": len(warehouse["reconciliation"]),
                           "all_passed": warehouse["all_checks_passed"]},
        "forecast": forecast,
        "seasonal": {k: v for k, v in profile.items() if k != "seasonal_component"},
        "zones_by_volume": by_volume,
        "zones_by_value": by_value,
        "volume_value_overlap": len(volume_names & value_names),
        "backtest": backtest,
        "tests": test_line,
        "seconds": round(time.time() - started, 1),
    }
    (OUTPUTS / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    conn.close()
    print(f"\n  tests: {test_line}")
    print(f"  wrote outputs/results.json ({report['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
