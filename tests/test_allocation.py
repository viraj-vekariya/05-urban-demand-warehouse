"""The allocation algorithm: the bounded heap, the capacity constraint, and the policies.

Property tests against constructions where the right answer is known by hand, not
regression tests against saved output.
"""

import random

import pytest

from src.allocate import (Cell, allocate_by_volume, allocate_day, allocate_hour,
                          allocate_uniform, top_k_cells)


def _cells(n=500, hour=9, dow=1, seed=7):
    rng = random.Random(seed)
    return [Cell(zone_id=i, zone_name=f"z{i}", hour=hour, dow=dow,
                 expected_trips_per_hour=rng.uniform(1, 60),
                 revenue_per_driver_hour=rng.uniform(20, 130),
                 borough="Manhattan") for i in range(n)]


# -- the heap ----------------------------------------------------------------

def test_the_heap_returns_exactly_what_a_full_sort_would():
    """The whole justification for the heap is that it is cheaper, not different. If it
    disagreed with a sort, it would be a bug wearing an optimisation's clothes."""
    cells = _cells(1000)
    top, _ = top_k_cells(cells, 40)
    expected = sorted(cells, key=lambda c: -c.revenue_per_driver_hour)[:40]
    assert [c.zone_id for c in top] == [c.zone_id for c in expected]


def test_the_heap_is_bounded_by_k_not_by_n():
    """O(n log k) with O(k) memory. Operation count must scale with k, and must be far
    below n once the heap is full and most cells are discarded in O(1)."""
    cells = _cells(5000)
    _, ops_small = top_k_cells(cells, 10)
    _, ops_large = top_k_cells(cells, 1000)
    assert ops_small < ops_large
    assert ops_small < len(cells) / 2, "the heap is not rejecting cheap candidates"


def test_the_heap_handles_exact_ties_without_crashing():
    """Equal keys make Python compare the payload objects, which raises TypeError unless
    a tie-breaker is inserted. Real data has ties."""
    cells = [Cell(i, f"z{i}", 9, 1, 10.0, 50.0) for i in range(50)]
    top, _ = top_k_cells(cells, 10)
    assert len(top) == 10


def test_asking_for_more_than_exists_returns_everything():
    cells = _cells(5)
    top, _ = top_k_cells(cells, 100)
    assert len(top) == 5


def test_asking_for_zero_returns_nothing():
    assert top_k_cells(_cells(10), 0) == ([], 0)


# -- capacity ----------------------------------------------------------------

def test_capacity_scales_with_observed_demand():
    """A cell that saw 40 trips can absorb more drivers than one that saw 2. Without
    this the allocation sends the whole fleet to one cell."""
    busy = Cell(1, "busy", 9, 1, expected_trips_per_hour=40, revenue_per_driver_hour=50)
    quiet = Cell(2, "quiet", 9, 1, expected_trips_per_hour=2, revenue_per_driver_hour=50)
    assert busy.capacity() > quiet.capacity()


def test_capacity_is_never_zero():
    """A cell with a trickle of demand should take one driver, not be unassignable."""
    assert Cell(1, "z", 9, 1, expected_trips_per_hour=0.01,
                revenue_per_driver_hour=50).capacity() == 1


def test_no_cell_is_assigned_beyond_its_capacity():
    """The constraint that makes this a fill rather than a sort."""
    result = allocate_hour(_cells(200), fleet_size=800, hour=9, dow=1)
    for assignment in result.assignments:
        assert assignment.drivers <= assignment.capacity


def test_a_large_fleet_spreads_across_many_cells():
    """The naive top-k failure mode: everyone to the single best cell."""
    small = allocate_hour(_cells(200), fleet_size=20, hour=9, dow=1)
    large = allocate_hour(_cells(200), fleet_size=1500, hour=9, dow=1)
    assert large.cells_used > small.cells_used


# -- the allocation ----------------------------------------------------------

def test_drivers_are_conserved():
    """placed + idle must equal the fleet. A policy that loses drivers is reporting
    revenue for people who were never sent anywhere."""
    for fleet in (10, 100, 1000, 5000):
        result = allocate_hour(_cells(200), fleet_size=fleet, hour=9, dow=1)
        assert result.drivers_placed + result.drivers_idle == fleet


def test_assignments_are_ordered_by_value():
    result = allocate_hour(_cells(300), fleet_size=200, hour=9, dow=1)
    rates = [a.revenue_per_driver_hour for a in result.assignments]
    assert rates == sorted(rates, reverse=True)


def test_only_the_requested_hour_and_day_are_considered():
    cells = _cells(100, hour=9, dow=1) + _cells(100, hour=17, dow=1, seed=8)
    result = allocate_hour(cells, fleet_size=50, hour=9, dow=1)
    assert all(a.hour == 9 for a in result.assignments)


def test_an_empty_hour_leaves_every_driver_idle():
    result = allocate_hour(_cells(50, hour=9), fleet_size=100, hour=3, dow=1)
    assert result.drivers_placed == 0 and result.drivers_idle == 100


def test_zero_revenue_cells_are_never_assigned():
    cells = [Cell(1, "dead", 9, 1, 50.0, 0.0), Cell(2, "live", 9, 1, 50.0, 40.0)]
    result = allocate_hour(cells, fleet_size=10, hour=9, dow=1)
    assert all(a.zone_id != 1 for a in result.assignments)


def test_a_full_day_covers_every_hour():
    day = allocate_day([c for h in range(24) for c in _cells(30, hour=h, dow=2, seed=h)],
                       fleet_size=100, dow=2)
    assert len(day["hours"]) == 24


# -- policy comparison -------------------------------------------------------

def test_revenue_ranking_beats_volume_ranking_on_expected_revenue():
    """The project's central claim, on data where it must hold by construction: the
    revenue policy optimises the quantity being scored."""
    cells = _cells(400)
    revenue = allocate_hour(cells, 300, 9, 1).expected_revenue
    volume = allocate_by_volume(cells, 300, 9, 1).expected_revenue
    assert revenue > volume


def test_revenue_ranking_beats_a_uniform_spread():
    cells = _cells(400)
    assert (allocate_hour(cells, 300, 9, 1).expected_revenue
            > allocate_uniform(cells, 300, 9, 1).expected_revenue)


def test_volume_ranking_really_does_chase_trip_count():
    """Confirms the baseline is what it claims to be - otherwise the comparison is
    against a straw man rather than the policy people actually use."""
    cells = [Cell(1, "many-cheap", 9, 1, 100.0, 20.0),
             Cell(2, "few-rich", 9, 1, 5.0, 200.0)]
    top = allocate_by_volume(cells, 10, 9, 1).assignments[0]
    assert top.zone_name == "many-cheap"
    assert allocate_hour(cells, 10, 9, 1).assignments[0].zone_name == "few-rich"
