"""Trip-level quality rules, and the reasons behind each one.

Raw TLC data is genuinely dirty in ways that matter for this analysis specifically:
negative fares from refunds, trips timestamped years outside their own file, zero-second
trips, and a location code (264/265) that means "unknown zone" rather than a place.

Every rule below classifies rather than deletes. The counts must reconcile to the source,
and a rule that silently removed 8% of a month would otherwise be invisible - which is
exactly how a demand model ends up trained on a biased subset without anyone noticing.

The rules are ordered by severity so a trip failing several is labelled by the most
serious, which makes the categories partition the data instead of overlapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# TLC's own data dictionary: 264 and 265 are "Unknown" and "NA". They are not zones, and
# aggregating demand into them would create two phantom hotspots.
UNKNOWN_ZONES = (264, 265)

# A yellow cab is metered; a 0-second or 0-mile trip is a meter error or a cancelled
# ride, not a journey anyone was carried on.
MIN_DURATION_SEC = 60
MAX_DURATION_SEC = 6 * 3600          # 6 hours; longer is a meter left running
MIN_DISTANCE_MI = 0.1
MAX_DISTANCE_MI = 100.0              # longer than Manhattan to Montauk
MIN_FARE = 2.50                      # TLC's own minimum metered fare
MAX_FARE = 1000.0
MAX_PASSENGERS = 6


@dataclass
class QualityRule:
    name: str
    predicate: str                   # SQL, applied in order
    reason: str


RULES: List[QualityRule] = [
    QualityRule(
        "null_timestamps",
        "tpep_pickup_datetime IS NULL OR tpep_dropoff_datetime IS NULL",
        "a trip with no time cannot be placed in an hour-of-week cell, which is the "
        "unit of analysis"),
    QualityRule(
        "wrong_month",
        "DATE_TRUNC('month', tpep_pickup_datetime) <> DATE '{month}'",
        "TLC files contain a small number of trips timestamped in other years - meter "
        "clock errors. Left in, they create demand in months that have no data"),
    QualityRule(
        "negative_duration",
        "tpep_dropoff_datetime <= tpep_pickup_datetime",
        "dropoff before pickup; a clock or data-entry error"),
    QualityRule(
        "implausible_duration",
        f"DATE_DIFF('second', tpep_pickup_datetime, tpep_dropoff_datetime) < {MIN_DURATION_SEC} "
        f"OR DATE_DIFF('second', tpep_pickup_datetime, tpep_dropoff_datetime) > {MAX_DURATION_SEC}",
        "under a minute is a meter error or cancellation; over six hours is a meter "
        "left running"),
    QualityRule(
        "unknown_zone",
        f"PULocationID IN {UNKNOWN_ZONES} OR DOLocationID IN {UNKNOWN_ZONES}",
        "264 and 265 are TLC's 'Unknown' and 'NA' codes, not places. Aggregating into "
        "them would create two phantom hotspots that the allocation would then chase"),
    QualityRule(
        "implausible_distance",
        f"trip_distance < {MIN_DISTANCE_MI} OR trip_distance > {MAX_DISTANCE_MI}",
        "zero-mile trips are meter errors; 100+ mile trips are out of the service area"),
    QualityRule(
        "non_positive_fare",
        f"total_amount < {MIN_FARE}",
        "refunds and disputes appear as negative totals. They are real events but they "
        "are not demand, and averaging them into revenue-per-hour understates every cell"),
    QualityRule(
        "implausible_fare",
        f"total_amount > {MAX_FARE}",
        "a four-figure metered fare is an error, and a handful of them dominate any mean"),
    QualityRule(
        "implausible_passengers",
        f"passenger_count > {MAX_PASSENGERS}",
        "a yellow cab seats at most six; a larger count is a data-entry error. "
        "NOTE what this rule deliberately does NOT exclude: NULL and zero. An earlier "
        "version treated a missing passenger_count as disqualifying, and it was the "
        "single largest exclusion in the whole pipeline - 12.66 percentage points of a "
        "17.97% exclusion rate, 483,731 trips in one month. Investigating them showed "
        "89.5% were otherwise completely valid trips: real fares, real zones, real "
        "durations. Some TLC vendors simply stopped reporting the field. Worse, they "
        "were not a random slice - their mean distance was 20.11 miles against a fleet "
        "average near 3, i.e. disproportionately airport runs. Dropping them would have "
        "removed an eighth of all demand AND systematically understated revenue in "
        "exactly the high-value zones the allocation exists to find. passenger_count is "
        "not used to compute demand or revenue, so a missing value is no reason to "
        "discard a trip"),
]


def build_case_expression(month: str) -> str:
    """A single CASE that assigns exactly one status per row.

    One expression rather than nine passes: DuckDB evaluates it once per row, and more
    importantly the WHEN order guarantees a partition. Nine separate boolean columns
    would let a row be counted in several buckets and the reconciliation would fail for
    a reason that has nothing to do with data quality.
    """
    branches = "\n        ".join(
        f"WHEN {rule.predicate.format(month=month)} THEN '{rule.name}'"
        for rule in RULES)
    return f"CASE\n        {branches}\n        ELSE 'ok'\n    END"


def rule_documentation() -> List[Dict[str, str]]:
    return [{"rule": r.name, "reason": r.reason} for r in RULES]
