"""Run the SQL chain over the partitioned Parquet, and enforce reconciliation.

Run:  python3 -m src.build
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"
DATA = ROOT / "data"
WAREHOUSE = DATA / "warehouse"
DB_PATH = DATA / "urban.duckdb"
OUTPUTS = ROOT / "outputs"


def connect() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(f"SET VARIABLE zone_path = '{(DATA / 'raw' / 'taxi_zone_lookup.csv').as_posix()}'")
    # A glob across partitions rather than a list of files: adding a month means dropping
    # a file into the directory, not editing SQL.
    conn.execute(f"SET VARIABLE warehouse_glob = '{(WAREHOUSE / '*' / '*.parquet').as_posix()}'")
    # DuckDB defaults to all cores; capped so the build does not starve anything else on
    # a laptop and so timings are comparable between runs.
    conn.execute("SET threads = 4")
    return conn


def main() -> int:
    if not WAREHOUSE.exists() or not any(WAREHOUSE.glob("*/*.parquet")):
        print("no partitions; run: python3 -m etl.incremental", file=sys.stderr)
        return 1

    conn = connect()
    print("building the demand warehouse")
    steps = []
    for path in sorted(SQL_DIR.glob("*.sql")):
        started = time.perf_counter()
        conn.execute(path.read_text())
        ms = round((time.perf_counter() - started) * 1000, 1)
        steps.append({"file": path.name, "ms": ms})
        print(f"  {path.name:<24} {ms:>9,.1f} ms")

    print("\nreconciliation")
    checks = []
    for name, left, right in conn.execute(
            "SELECT check_name, left_value, right_value FROM reconciliation").fetchall():
        passed = int(left) == int(right)
        checks.append({"check": name, "left": int(left), "right": int(right),
                       "passed": passed})
        print(f"  {'ok ' if passed else 'FAIL'} {name:<38} "
              f"{int(left):>14,} vs {int(right):>14,}")

    failed = [c for c in checks if not c["passed"]]

    cells = conn.execute("SELECT COUNT(*) FROM mart_demand_profile").fetchone()[0]
    trips, revenue, hours = conn.execute(
        "SELECT SUM(trips), SUM(revenue), SUM(driver_hours) FROM mart_demand").fetchone()
    months = conn.execute("SELECT COUNT(*) FROM mart_monthly").fetchone()[0]

    print(f"\n  {trips:,} trips  ${revenue:,.0f} revenue  {hours:,.0f} driver-hours")
    print(f"  {cells:,} zone x dow x hour cells across {months} months")

    top = conn.execute("""
        SELECT zone_name, borough, trips, revenue_per_driver_hour
        FROM mart_zone_summary WHERE trips > 20000
        ORDER BY revenue_per_driver_hour DESC LIMIT 6""").fetchall()
    print("\n  highest revenue per driver-hour (zones with >20k trips)")
    for name, borough, n, rph in top:
        print(f"    {name[:34]:<34} {borough:<10} {n:>9,} trips  ${rph:>7.2f}/hr")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sql_steps": steps,
        "reconciliation": checks,
        "all_checks_passed": not failed,
        "totals": {"trips": int(trips), "revenue": float(revenue),
                   "driver_hours": float(hours), "cells": int(cells),
                   "months": int(months)},
        "monthly": [{"month": m, "trips": int(t), "revenue": float(r),
                     "revenue_per_driver_hour": float(rph)}
                    for m, t, r, _, rph in conn.execute(
                        "SELECT month, trips, revenue, driver_hours, "
                        "revenue_per_driver_hour FROM mart_monthly").fetchall()],
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "warehouse.json").write_text(json.dumps(summary, indent=2) + "\n")
    conn.close()

    if failed:
        print(f"\n  {len(failed)} RECONCILIATION CHECK(S) FAILED", file=sys.stderr)
        return 1
    print(f"\n  {len(checks)}/{len(checks)} reconciliation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
