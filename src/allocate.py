"""Fleet allocation: where to position supply, under a hard fleet constraint.

THE DECISION. Given N drivers and a forecast of demand for each (zone, hour) cell
tomorrow, which cells should they be sent to?

THE ALGORITHM. For each hour independently, rank cells by expected revenue per driver-hour
and take the top k under the fleet constraint, using a BOUNDED MIN-HEAP.

Why a bounded heap rather than sorting everything:
  * sorting all 44,520 cells is O(n log n) and produces a full ordering nobody reads;
  * a heap of size k is O(n log k) and, critically, O(k) MEMORY. With k = 500 drivers that
    is a 500-element heap regardless of how many candidate cells exist.
  * On this data the difference in wall-clock is small - 44k items is not many. The reason
    it is the right structure anyway is that the candidate set scales with zones x
    granularity, and the fleet does not. A city with per-block granularity has millions of
    cells and 500 drivers, and there the distinction stops being academic.

THE CONSTRAINT THAT MAKES IT NON-TRIVIAL. A naive top-k sends every driver to the single
best cell. That is wrong for a reason the data itself shows: a cell that historically
served 40 trips an hour cannot absorb 200 drivers, and the 201st driver there earns
nothing. So each cell has a CAPACITY derived from its own observed demand, and the
allocation fills cells in value order subject to capacity - which turns a sort into a
greedy knapsack-style fill.

WHAT THIS IS NOT. It is not an optimal assignment. Optimal would require modelling
deadheading between cells, driver locations, and the fact that a driver sent to JFK ends
up somewhere else afterwards - a min-cost-flow over a time-expanded network. The greedy
fill is the honest 90% solution, and src/backtest.py measures how much it actually earns
against alternatives rather than assuming it wins.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Cell:
    """One (zone, hour) opportunity."""

    zone_id: int
    zone_name: str
    hour: int
    dow: int
    expected_trips_per_hour: float
    revenue_per_driver_hour: float
    borough: str = ""

    def capacity(self, drivers_per_trip: float = 1.0,
                 max_multiple: float = 1.5) -> int:
        """How many drivers this cell can absorb in an hour.

        Derived from the cell's own observed trip rate rather than assumed. The
        `max_multiple` headroom above historical demand exists because the forecast is
        not exact and a cell that averaged 40 trips can plausibly see 60 - but it is
        bounded, because sending 200 drivers to a 40-trip cell is how a fleet ends up
        idle in one place while the rest of the city goes unserved.
        """
        return max(1, int(self.expected_trips_per_hour * drivers_per_trip * max_multiple))


@dataclass
class Assignment:
    zone_id: int
    zone_name: str
    borough: str
    hour: int
    drivers: int
    revenue_per_driver_hour: float
    expected_revenue: float
    capacity: int
    utilisation: float

    def as_dict(self) -> Dict[str, object]:
        return {"zone_id": self.zone_id, "zone_name": self.zone_name,
                "borough": self.borough, "hour": self.hour, "drivers": self.drivers,
                "revenue_per_driver_hour": round(self.revenue_per_driver_hour, 2),
                "expected_revenue": round(self.expected_revenue, 2),
                "capacity": self.capacity, "utilisation": round(self.utilisation, 4)}


@dataclass
class AllocationResult:
    hour: int
    dow: int
    fleet_size: int
    assignments: List[Assignment]
    drivers_placed: int
    drivers_idle: int
    expected_revenue: float
    cells_considered: int
    cells_used: int
    heap_operations: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "hour": self.hour, "dow": self.dow, "fleet_size": self.fleet_size,
            "drivers_placed": self.drivers_placed, "drivers_idle": self.drivers_idle,
            "expected_revenue": round(self.expected_revenue, 2),
            "expected_revenue_per_driver": round(
                self.expected_revenue / self.fleet_size, 2) if self.fleet_size else 0.0,
            "cells_considered": self.cells_considered, "cells_used": self.cells_used,
            "heap_operations": self.heap_operations,
            "assignments": [a.as_dict() for a in self.assignments],
        }


def top_k_cells(cells: Iterable[Cell], k: int) -> Tuple[List[Cell], int]:
    """The k highest-value cells, via a bounded min-heap. Returns (cells, heap ops).

    The heap holds the k best seen so far, with the WORST of them at the root. A new cell
    is compared against the root: if it is not better than the worst kept, it is discarded
    in O(1) without ever entering the heap. That is why this is O(n log k) rather than
    O(n log n), and why memory is O(k).

    `heap_operations` is returned because it is the honest way to compare this against a
    full sort - wall clock on 44k items is dominated by noise, while operation counts are
    deterministic.
    """
    if k <= 0:
        return [], 0

    heap: List[Tuple[float, int, Cell]] = []
    operations = 0
    # A monotonically increasing tie-breaker. Without it, two cells with identical
    # revenue make Python compare the Cell objects themselves, which raises TypeError.
    # It also makes ties resolve deterministically by insertion order rather than by
    # whatever the heap happens to do.
    counter = 0

    for cell in cells:
        counter += 1
        key = cell.revenue_per_driver_hour
        if len(heap) < k:
            heapq.heappush(heap, (key, counter, cell))
            operations += 1
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, counter, cell))
            operations += 1
        # else: discarded in O(1) without touching the heap at all

    ordered = sorted(heap, key=lambda item: item[0], reverse=True)
    return [item[2] for item in ordered], operations


def allocate_hour(cells: Sequence[Cell], fleet_size: int, hour: int, dow: int,
                  max_multiple: float = 1.5) -> AllocationResult:
    """Greedy capacity-constrained fill for one hour.

    Cells are taken in descending revenue order and filled to capacity until the fleet is
    exhausted. Greedy is optimal for this particular problem - each driver-hour is
    identical and independent, and capacities are hard - which is worth saying because
    greedy is usually a heuristic and here it is not.
    """
    candidates = [c for c in cells if c.hour == hour and c.dow == dow
                  and c.revenue_per_driver_hour > 0]

    # Fetch more cells than drivers: one high-value cell may only absorb a handful, so a
    # top-`fleet_size` fetch can run out of capacity before it runs out of drivers.
    fetch = min(len(candidates), max(fleet_size, 1) * 4)
    ranked, operations = top_k_cells(candidates, fetch)

    assignments: List[Assignment] = []
    remaining = fleet_size
    total_revenue = 0.0

    for cell in ranked:
        if remaining <= 0:
            break
        capacity = cell.capacity(max_multiple=max_multiple)
        drivers = min(remaining, capacity)
        if drivers <= 0:
            continue
        revenue = drivers * cell.revenue_per_driver_hour
        assignments.append(Assignment(
            zone_id=cell.zone_id, zone_name=cell.zone_name, borough=cell.borough,
            hour=hour, drivers=drivers,
            revenue_per_driver_hour=cell.revenue_per_driver_hour,
            expected_revenue=revenue, capacity=capacity,
            utilisation=drivers / capacity))
        total_revenue += revenue
        remaining -= drivers

    return AllocationResult(
        hour=hour, dow=dow, fleet_size=fleet_size, assignments=assignments,
        drivers_placed=fleet_size - remaining, drivers_idle=remaining,
        expected_revenue=total_revenue, cells_considered=len(candidates),
        cells_used=len(assignments), heap_operations=operations)


def allocate_day(cells: Sequence[Cell], fleet_size: int, dow: int,
                 max_multiple: float = 1.5) -> Dict[str, object]:
    """A full 24-hour schedule. Each hour is solved independently.

    Independence is an assumption, and a real one: it ignores that a driver who ends an
    hour at JFK starts the next hour at JFK. Modelling that turns this into a
    min-cost-flow over a time-expanded graph. It is stated here rather than hidden, and
    the backtest measures the cost of the simplification empirically.
    """
    hours = [allocate_hour(cells, fleet_size, h, dow, max_multiple) for h in range(24)]
    total_revenue = sum(h.expected_revenue for h in hours)
    return {
        "dow": dow,
        "fleet_size": fleet_size,
        "total_expected_revenue": round(total_revenue, 2),
        "revenue_per_driver_day": round(total_revenue / fleet_size, 2) if fleet_size else 0,
        "total_idle_driver_hours": sum(h.drivers_idle for h in hours),
        "hours": [h.as_dict() for h in hours],
    }


# -- baselines, so the allocation has something to beat --------------------------------

def allocate_uniform(cells: Sequence[Cell], fleet_size: int, hour: int,
                     dow: int) -> AllocationResult:
    """Spread the fleet evenly across every active cell. The 'do nothing clever' policy."""
    candidates = [c for c in cells if c.hour == hour and c.dow == dow
                  and c.revenue_per_driver_hour > 0]
    if not candidates:
        return AllocationResult(hour, dow, fleet_size, [], 0, fleet_size, 0.0, 0, 0, 0)

    per_cell = max(1, fleet_size // len(candidates))
    assignments, remaining, revenue = [], fleet_size, 0.0
    for cell in candidates:
        if remaining <= 0:
            break
        drivers = min(remaining, per_cell)
        assignments.append(Assignment(
            cell.zone_id, cell.zone_name, cell.borough, hour, drivers,
            cell.revenue_per_driver_hour, drivers * cell.revenue_per_driver_hour,
            cell.capacity(), drivers / max(1, cell.capacity())))
        revenue += drivers * cell.revenue_per_driver_hour
        remaining -= drivers
    return AllocationResult(hour, dow, fleet_size, assignments,
                            fleet_size - remaining, remaining, revenue,
                            len(candidates), len(assignments), 0)


def allocate_by_volume(cells: Sequence[Cell], fleet_size: int, hour: int,
                       dow: int, max_multiple: float = 1.5) -> AllocationResult:
    """Rank by TRIP COUNT instead of revenue per driver-hour.

    This is the policy the analysis exists to argue against, and it is the one most
    dashboards implicitly encourage - "send drivers where the demand is". It is included
    as a measured baseline rather than a straw man, because if it turned out to earn just
    as much, the whole revenue-per-driver-hour argument would be decoration.
    """
    candidates = [c for c in cells if c.hour == hour and c.dow == dow
                  and c.expected_trips_per_hour > 0]
    ranked = sorted(candidates, key=lambda c: c.expected_trips_per_hour, reverse=True)

    assignments, remaining, revenue = [], fleet_size, 0.0
    for cell in ranked:
        if remaining <= 0:
            break
        capacity = cell.capacity(max_multiple=max_multiple)
        drivers = min(remaining, capacity)
        assignments.append(Assignment(
            cell.zone_id, cell.zone_name, cell.borough, hour, drivers,
            cell.revenue_per_driver_hour, drivers * cell.revenue_per_driver_hour,
            capacity, drivers / capacity))
        revenue += drivers * cell.revenue_per_driver_hour
        remaining -= drivers
    return AllocationResult(hour, dow, fleet_size, assignments,
                            fleet_size - remaining, remaining, revenue,
                            len(candidates), len(assignments), 0)


POLICIES = {
    "revenue_per_hour": allocate_hour,
    "trip_volume": allocate_by_volume,
    "uniform": allocate_uniform,
}
