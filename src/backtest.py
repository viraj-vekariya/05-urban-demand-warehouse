"""Backtest the allocation policy against held-out months.

THE QUESTION. The allocation is built from historical cell profiles. Does following it
actually earn more than the alternatives, on months the profile has never seen?

THE DESIGN, and every element of it exists to stop this from being a rigged comparison:

  * **Temporal split, never random.** Train on months 1-9, test on 10-12. A random split
    would leak: the same week appears on both sides and the policy is evaluated on demand
    it was built from. Time series must be split by time.
  * **Three policies, one of which is the naive default.** revenue-per-driver-hour against
    trip-volume ranking and a uniform spread. If ranking by volume earned the same, the
    project's central claim would be decoration.
  * **Realised revenue, not expected.** A policy assigns drivers to cells using TRAINING
    profiles; the revenue they earn is computed from the TEST months' actual observed
    rates. Scoring a policy against the numbers it was built from measures nothing.
  * **A capacity ceiling on realised revenue too.** A cell that actually saw 30 trips
    cannot pay 200 drivers, no matter what the plan said. Without this, an over-committing
    policy scores brilliantly on paper.

Run:  python3 -m src.backtest
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocate import Cell, allocate_by_volume, allocate_hour, allocate_uniform  # noqa: E402

DB = ROOT / "data" / "urban.duckdb"
OUTPUTS = ROOT / "outputs"

TRAIN_MONTHS = [f"2024-{m:02d}" for m in range(1, 10)]
TEST_MONTHS = [f"2024-{m:02d}" for m in range(10, 13)]


def load_cells(conn, months: Sequence[str]) -> List[Cell]:
    """Cell profiles averaged over the given months."""
    placeholders = ", ".join(f"'{m}'" for m in months)
    rows = conn.execute(f"""
        SELECT
            d.zone_id, ANY_VALUE(d.zone_name), ANY_VALUE(d.borough),
            ANY_VALUE(d.is_airport), d.dow, d.hour,
            AVG(d.trips_per_day)  AS trips_per_day,
            SUM(d.revenue) / NULLIF(SUM(d.driver_hours), 0) AS rph
        FROM mart_demand d
        WHERE d.month IN ({placeholders})
        GROUP BY d.zone_id, d.dow, d.hour
        HAVING SUM(d.driver_hours) > 0
    """).fetchall()
    return [Cell(zone_id=int(z), zone_name=str(name), hour=int(h), dow=int(dw),
                 expected_trips_per_hour=float(tpd or 0.0),
                 revenue_per_driver_hour=float(rph or 0.0), borough=str(boro))
            for z, name, boro, _air, dw, h, tpd, rph in rows]


def realised_rates(conn, months: Sequence[str]) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    """What each cell ACTUALLY paid and absorbed in the test months."""
    placeholders = ", ".join(f"'{m}'" for m in months)
    rows = conn.execute(f"""
        SELECT zone_id, dow, hour,
               SUM(revenue) / NULLIF(SUM(driver_hours), 0) AS rph,
               AVG(trips_per_day) AS trips_per_day
        FROM mart_demand WHERE month IN ({placeholders})
        GROUP BY zone_id, dow, hour HAVING SUM(driver_hours) > 0
    """).fetchall()
    return {(int(z), int(dw), int(h)): {"rph": float(rph or 0.0),
                                        "trips": float(tpd or 0.0)}
            for z, dw, h, rph, tpd in rows}


def score(result, realised: Dict[Tuple[int, int, int], Dict[str, float]],
          max_multiple: float = 1.5) -> Dict[str, float]:
    """Realised revenue for one hour's assignment.

    Drivers sent to a cell that does not appear in the test months earn NOTHING. That is
    the correct treatment - the plan sent them somewhere with no demand - and it is what
    penalises a policy that over-fits to quiet training cells.
    """
    revenue, placed, wasted, uncovered = 0.0, 0, 0, 0
    for assignment in result.assignments:
        key = (assignment.zone_id, result.dow, result.hour)
        actual = realised.get(key)
        if actual is None or actual["rph"] <= 0:
            wasted += assignment.drivers
            continue
        # The realised capacity ceiling: a cell that saw 30 trips cannot pay 200 drivers.
        capacity = max(1, int(actual["trips"] * max_multiple))
        earning = min(assignment.drivers, capacity)
        uncovered += assignment.drivers - earning
        revenue += earning * actual["rph"]
        placed += assignment.drivers
    return {"realised_revenue": revenue, "drivers_assigned": placed,
            "drivers_to_dead_cells": wasted, "drivers_over_capacity": uncovered}


def backtest(fleet_sizes: Sequence[int] = (100, 300, 500, 1000)) -> Dict[str, object]:
    conn = duckdb.connect(str(DB), read_only=True)
    started = time.time()

    train_cells = load_cells(conn, TRAIN_MONTHS)
    realised = realised_rates(conn, TEST_MONTHS)
    print(f"  {len(train_cells):,} training cells (months {TRAIN_MONTHS[0]}..{TRAIN_MONTHS[-1]})")
    print(f"  {len(realised):,} test cells   (months {TEST_MONTHS[0]}..{TEST_MONTHS[-1]})")

    policies = {"revenue_per_hour": allocate_hour,
                "trip_volume": allocate_by_volume,
                "uniform": allocate_uniform}

    results: List[Dict[str, object]] = []
    for fleet in fleet_sizes:
        for name, fn in policies.items():
            revenue = 0.0
            wasted = over = assigned = 0
            for dow in range(7):
                for hour in range(24):
                    allocation = fn(train_cells, fleet, hour, dow)
                    s = score(allocation, realised)
                    revenue += s["realised_revenue"]
                    wasted += s["drivers_to_dead_cells"]
                    over += s["drivers_over_capacity"]
                    assigned += s["drivers_assigned"]
            results.append({
                "fleet_size": fleet, "policy": name,
                "realised_weekly_revenue": round(revenue, 2),
                "revenue_per_driver_hour": round(revenue / (fleet * 168), 4),
                "drivers_to_dead_cells": wasted,
                "drivers_over_capacity": over,
                "driver_hours_assigned": assigned,
            })
            print(f"  fleet {fleet:>5}  {name:<18} ${revenue:>14,.0f}  "
                  f"${revenue / (fleet * 168):>7.2f}/driver-hour  "
                  f"dead {wasted:>6,}  over-capacity {over:>7,}")

    conn.close()

    by_fleet: Dict[int, Dict[str, float]] = {}
    for row in results:
        by_fleet.setdefault(row["fleet_size"], {})[row["policy"]] = \
            row["realised_weekly_revenue"]

    lifts = []
    for fleet, policies_revenue in by_fleet.items():
        best_baseline = max(policies_revenue["trip_volume"], policies_revenue["uniform"])
        lifts.append({
            "fleet_size": fleet,
            "revenue_policy": policies_revenue["revenue_per_hour"],
            "best_baseline": best_baseline,
            "best_baseline_policy": ("trip_volume"
                                     if policies_revenue["trip_volume"] >= policies_revenue["uniform"]
                                     else "uniform"),
            "lift_pct": round(100 * (policies_revenue["revenue_per_hour"] - best_baseline)
                              / best_baseline, 2) if best_baseline else None,
            "vs_uniform_pct": round(100 * (policies_revenue["revenue_per_hour"]
                                           - policies_revenue["uniform"])
                                    / policies_revenue["uniform"], 2)
            if policies_revenue["uniform"] else None,
        })

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_months": TRAIN_MONTHS, "test_months": TEST_MONTHS,
        "train_cells": len(train_cells), "test_cells": len(realised),
        "results": results,
        "lift": lifts,
        "mean_lift_over_best_baseline_pct": round(
            statistics.fmean([l["lift_pct"] for l in lifts if l["lift_pct"] is not None]), 2),
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    if not DB.exists():
        print("no warehouse; run python3 -m src.build", file=sys.stderr)
        return 1
    print(f"backtest: train {TRAIN_MONTHS[0]}..{TRAIN_MONTHS[-1]}, "
          f"test {TEST_MONTHS[0]}..{TEST_MONTHS[-1]}\n")
    report = backtest()

    print(f"\n  lift of revenue-per-hour ranking over the best baseline")
    for row in report["lift"]:
        print(f"    fleet {row['fleet_size']:>5}: {row['lift_pct']:>+7.2f}% "
              f"(vs {row['best_baseline_policy']}), "
              f"{row['vs_uniform_pct']:>+7.2f}% vs uniform")
    print(f"\n  mean lift over the best baseline: "
          f"{report['mean_lift_over_best_baseline_pct']:+.2f}%")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "backtest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"  wrote outputs/backtest.json ({report['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
