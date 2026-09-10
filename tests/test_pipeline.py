"""ETL quality rules, seasonal decomposition, and the forecast's honesty checks."""

import math

import numpy as np
import pytest

from etl.quality import RULES, UNKNOWN_ZONES, build_case_expression, rule_documentation
from src.forecast import (build_features, cyclical, historical_mean_baseline, metrics)
from src.seasonal import decompose, hour_of_week_profile, moving_average


# -- quality rules -----------------------------------------------------------

def test_every_rule_documents_its_reason():
    """A quality rule without a stated reason is a rule nobody can challenge, and
    quality rules silently bias analyses."""
    for rule in RULES:
        assert rule.reason and len(rule.reason) > 20, f"{rule.name} has no real reason"


def test_the_case_expression_is_a_single_partition():
    """One CASE with ordered WHENs, so a row failing several rules is labelled once.
    Separate boolean columns would double-count and the reconciliation would fail for a
    reason unrelated to data quality."""
    expr = build_case_expression("2024-01-01")
    assert expr.count("CASE") == 1
    assert expr.count("WHEN") == len(RULES)
    assert expr.strip().endswith("END")


def test_missing_passenger_count_is_not_an_exclusion():
    """REGRESSION, and the most consequential bug found in this project. Treating a NULL
    passenger_count as disqualifying excluded 12.66 percentage points of a 17.97%
    exclusion rate - 483,731 trips in one month, 89.5% of which were otherwise valid,
    with a mean distance of 20.11 miles (disproportionately airport runs). It removed an
    eighth of all demand AND biased it toward exactly the high-value zones the
    allocation exists to find. Fixing it took exclusions from 14.71% to 4.78%."""
    rule = [r for r in RULES if r.name == "implausible_passengers"][0]
    assert "IS NULL" not in rule.predicate
    assert "< 1" not in rule.predicate
    assert "> 6" in rule.predicate


def test_unknown_zones_are_excluded():
    """264 and 265 are TLC's 'Unknown' and 'NA' codes, not places. Aggregating into them
    creates two phantom hotspots the allocation would chase."""
    rule = [r for r in RULES if r.name == "unknown_zone"][0]
    assert all(str(z) in rule.predicate for z in UNKNOWN_ZONES)


def test_the_month_filter_is_parameterised():
    """TLC files contain trips timestamped in other years from meter clock errors."""
    expr = build_case_expression("2024-03-01")
    assert "2024-03-01" in expr


# -- seasonal ----------------------------------------------------------------

def test_moving_average_returns_none_where_the_window_does_not_fit():
    """None rather than padding: inventing endpoints makes the trend look confident
    exactly where there is least information."""
    result = moving_average([1, 2, 3, 4, 5], 3)
    assert result[0] is None and result[-1] is None
    assert result[2] == pytest.approx(3.0)


def test_an_even_window_uses_a_2xM_average():
    result = moving_average([1.0] * 10, 4)
    assert all(v == pytest.approx(1.0) for v in result if v is not None)


def test_decomposition_recovers_a_planted_seasonal_pattern():
    period = 24
    seasonal = [math.sin(2 * math.pi * i / period) * 10 for i in range(period)]
    series = [100 + seasonal[i % period] for i in range(period * 6)]
    result = decompose(series, period)
    for i in range(period):
        assert result.seasonal[i] == pytest.approx(seasonal[i], abs=0.5)


def test_seasonal_components_sum_to_zero():
    """Uncentred components absorb part of the level and bias the trend."""
    series = [100 + (i % 24) for i in range(24 * 5)]
    result = decompose(series, 24)
    assert sum(result.seasonal[:24]) == pytest.approx(0.0, abs=1e-6)


def test_pure_noise_has_low_seasonal_strength():
    """The metric must be able to say 'there is no cycle here', or it says nothing."""
    rng = np.random.default_rng(5)
    series = list(rng.normal(100, 10, 24 * 8))
    assert decompose(series, 24).strength()["seasonal_strength"] < 0.6


def test_too_short_a_series_is_refused():
    with pytest.raises(ValueError, match="two full periods"):
        decompose([1.0] * 10, 24)


def test_hour_of_week_profile_labels_the_peak_readably():
    series = ([50.0] * 168) * 3
    series[100] = 500.0
    series[268] = 500.0
    series[436] = 500.0
    profile = hour_of_week_profile(series)
    assert profile["peak_hour_of_week"] == 100
    assert "Thursday" in profile["peak_label"] or ":" in profile["peak_label"]


# -- forecast ----------------------------------------------------------------

def test_cyclical_encoding_puts_hour_23_next_to_hour_0():
    """The reason for sin/cos rather than raw integers: a model given integers has to
    spend splits learning that 23 is adjacent to 0."""
    sin, cos = cyclical(np.array([0.0, 23.0, 12.0]), 24)
    d_wrap = math.hypot(sin[0] - sin[1], cos[0] - cos[1])
    d_far = math.hypot(sin[0] - sin[2], cos[0] - cos[2])
    assert d_wrap < d_far


def test_mape_ignores_zero_actuals():
    """Many cells are genuinely zero. Dividing by them produces infinities that swamp
    the metric."""
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([1.0, 11.0, 19.0])
    result = metrics(actual, predicted, "t")
    assert result.mape is not None and math.isfinite(result.mape)


def test_a_perfect_prediction_scores_perfectly():
    actual = np.array([1.0, 5.0, 9.0])
    result = metrics(actual, actual.copy(), "t")
    assert result.mae == 0 and result.rmse == 0 and result.r2 == pytest.approx(1.0)


def test_the_baseline_falls_back_for_unseen_cells():
    """A baseline that skipped hard rows would not be a fair comparison."""
    train = [{"zone_id": 1, "dow": 1, "hour": 9, "target": 10.0}]
    test = [{"zone_id": 99, "dow": 3, "hour": 4, "target": 5.0}]
    prediction = historical_mean_baseline(train, test)
    assert len(prediction) == 1 and np.isfinite(prediction[0])


def test_the_baseline_uses_the_cell_mean_when_it_has_one():
    train = [{"zone_id": 1, "dow": 1, "hour": 9, "target": 10.0},
             {"zone_id": 1, "dow": 1, "hour": 9, "target": 20.0}]
    test = [{"zone_id": 1, "dow": 1, "hour": 9, "target": 0.0}]
    assert historical_mean_baseline(train, test)[0] == pytest.approx(15.0)


def test_features_have_a_stable_shape_and_no_nans():
    rows = [{"zone_id": 1, "dow": d, "hour": h, "target": 1.0}
            for d in range(7) for h in range(24)]
    X, names = build_features(rows)
    assert X.shape == (len(rows), len(names))
    assert np.isfinite(X).all()
