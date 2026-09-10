"""The SQL warehouse: reconciliation, the metric definitions, and the spine."""

from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "urban.duckdb"

pytestmark = pytest.mark.skipif(
    not DB.exists(), reason="no warehouse; run python3 -m etl.incremental && python3 -m src.build")


@pytest.fixture(scope="module")
def conn():
    c = duckdb.connect(str(DB), read_only=True)
    c.execute(f"SET VARIABLE warehouse_glob = "
              f"'{(ROOT / 'data' / 'warehouse' / '*' / '*.parquet').as_posix()}'")
    yield c
    c.close()


def test_every_reconciliation_check_passes(conn):
    rows = conn.execute("SELECT check_name, left_value, right_value "
                        "FROM reconciliation").fetchall()
    assert rows
    failures = [(n, l, r) for n, l, r in rows if int(l) != int(r)]
    assert not failures, f"reconciliation failed: {failures}"


def test_the_hour_of_week_spine_is_complete(conn):
    """168 cells, generated not derived. Deriving it from observed trips would silently
    drop the quiet hours - and 'nobody hails a cab here at 4am' is exactly what an
    allocation needs to know."""
    n = conn.execute("SELECT COUNT(*) FROM dim_time").fetchone()[0]
    assert n == 168
    distinct = conn.execute(
        "SELECT COUNT(DISTINCT hour_of_week) FROM dim_time").fetchone()[0]
    assert distinct == 168


def test_no_unknown_zones_reached_the_marts(conn):
    """264 and 265 are TLC's Unknown/NA codes. In the mart they would be two phantom
    hotspots the allocation would chase."""
    n = conn.execute(
        "SELECT COUNT(*) FROM mart_demand WHERE zone_id IN (264, 265)").fetchone()[0]
    assert n == 0


def test_revenue_per_driver_hour_is_computed_from_sums_not_averaged_ratios(conn):
    """An average of per-month ratios weights a month with 3 trips the same as one with
    30,000, which makes tiny cells look enormously profitable. The profile must divide
    total revenue by total hours."""
    rows = conn.execute("""
        SELECT p.revenue_per_driver_hour,
               SUM(d.revenue) / NULLIF(SUM(d.driver_hours), 0)
        FROM mart_demand_profile p
        JOIN mart_demand d ON d.zone_id = p.zone_id AND d.dow = p.dow AND d.hour = p.hour
        GROUP BY p.zone_id, p.dow, p.hour, p.revenue_per_driver_hour
        LIMIT 200
    """).fetchall()
    compared = [(a, b) for a, b in rows if a and b]
    assert compared, "no comparable cells were returned"
    mismatches = [(a, b) for a, b in compared if abs(a - b) > 1e-6]
    assert not mismatches, f"profile ratio disagrees with the recomputed ratio: {mismatches[:3]}"


def test_revenue_per_driver_hour_is_plausible(conn):
    """A sanity band. A metered yellow cab earning $2,000/hour is a data bug, not a
    finding, and the allocation would send the entire fleet to it."""
    row = conn.execute("""
        SELECT MIN(revenue_per_driver_hour), MAX(revenue_per_driver_hour)
        FROM mart_zone_summary WHERE trips > 10000""").fetchone()
    assert row[0] > 5, f"a busy zone earning ${row[0]:.2f}/hr is implausible"
    assert row[1] < 500, f"a busy zone earning ${row[1]:.2f}/hr is implausible"


def test_airports_are_the_highest_value_zones(conn):
    """A domain sanity check with a known answer: JFK and LaGuardia runs are long and
    expensive, so they must top revenue-per-driver-hour. If they did not, the metric is
    computed wrong."""
    top = conn.execute("""
        SELECT zone_name FROM mart_zone_summary
        WHERE trips > 20000 ORDER BY revenue_per_driver_hour DESC LIMIT 5""").fetchall()
    names = " ".join(n for (n,) in top)
    assert "Airport" in names, f"no airport in the top 5 by value: {names}"


def test_the_busiest_zone_is_not_the_most_valuable(conn):
    """The finding that motivates the whole allocation metric. If volume and value
    ranked the same, ranking by revenue would buy nothing."""
    busiest = conn.execute(
        "SELECT zone_name FROM mart_zone_summary ORDER BY trips DESC LIMIT 1").fetchone()[0]
    most_valuable = conn.execute(
        "SELECT zone_name FROM mart_zone_summary WHERE trips > 20000 "
        "ORDER BY revenue_per_driver_hour DESC LIMIT 1").fetchone()[0]
    assert busiest != most_valuable


def test_trips_per_day_normalises_for_month_length(conn):
    """A month has 4 or 5 of each weekday. Comparing raw totals would rank a 5-Monday
    month above a 4-Monday one for no real reason."""
    rows = conn.execute("""
        SELECT trips, days_observed, trips_per_day FROM mart_demand
        WHERE days_observed > 0 LIMIT 500""").fetchall()
    assert all(abs(tpd - t / d) < 1e-9 for t, d, tpd in rows)


def test_demand_has_a_believable_rush_hour(conn):
    """Domain check: weekday evening must beat 4am. If it does not, the hour extraction
    or the timezone is wrong - and a whole allocation would be built on shifted hours."""
    rows = dict(conn.execute("""
        SELECT hour, SUM(trips) FROM mart_demand WHERE dow BETWEEN 1 AND 5
        GROUP BY hour""").fetchall())
    assert rows[18] > rows[4] * 5, "the evening peak is missing or the hours are shifted"


def test_every_loaded_partition_reaches_the_mart(conn):
    """The real property is CONSISTENCY, not a fixed month count.

    An earlier version asserted exactly 12 months. That is an assumption about the
    environment, not about the pipeline: CI loads 3 months to keep the job fast, and the
    test failed on a build with nothing wrong in it. What actually matters is that every
    month the watermark says it loaded is present in the mart, and vice versa - which
    holds at any month count.
    """
    watermarked = {m for (m,) in conn.execute(
        "SELECT DISTINCT month FROM load_watermark").fetchall()}
    in_mart = {m for (m,) in conn.execute(
        "SELECT DISTINCT month FROM mart_demand").fetchall()}
    assert watermarked, "no partitions were loaded at all"
    assert watermarked == in_mart, (
        f"loaded but missing from the mart: {sorted(watermarked - in_mart)}; "
        f"in the mart but never loaded: {sorted(in_mart - watermarked)}")


def test_every_watermark_row_records_a_real_load(conn):
    n = conn.execute("SELECT COUNT(*) FROM load_watermark").fetchone()[0]
    assert n > 0, "the watermark table is empty"
    incomplete = conn.execute(
        "SELECT COUNT(*) FROM load_watermark WHERE loaded_rows <= 0").fetchone()[0]
    assert incomplete == 0, "a partition was recorded as loaded with zero rows"
    # Every partition must have kept the large majority of its source rows. A partition
    # that lost most of its trips would pass the consistency check above while being
    # badly broken.
    thin = conn.execute(
        "SELECT COUNT(*) FROM load_watermark WHERE loaded_rows < source_rows * 0.8"
    ).fetchone()[0]
    assert thin == 0, "a partition kept under 80% of its source rows"


def test_the_exclusion_rate_is_low_after_the_passenger_fix(conn):
    """REGRESSION guard for the biggest bug in the project. Treating NULL passenger_count
    as disqualifying pushed exclusions to 14.71%; the correct rule gives 4.78%. If this
    climbs back above 10%, that rule has been reintroduced."""
    row = conn.execute(
        "SELECT SUM(source_rows), SUM(loaded_rows) FROM load_watermark").fetchone()
    excluded_pct = 100.0 * (row[0] - row[1]) / row[0]
    assert excluded_pct < 10.0, (
        f"{excluded_pct:.2f}% of trips excluded - check whether a NULL passenger_count "
        f"is disqualifying again")
