"""Seasonal decomposition of demand: trend, weekly and daily components.

WHY THIS IS HERE AND NOT A MODEL. Before forecasting anything it is worth knowing how much
of the variation is simple periodicity. If the hour-of-week pattern explains most of it,
a gradient-boosted model is a very expensive way to learn "Friday evening is busy", and
the honest baseline for the forecast is the historical hour-of-week mean.

The decomposition is classical additive - trend by centred moving average, seasonal by
averaging the detrended series within each period, residual by subtraction. Written out
rather than imported because the period choices below are the substantive decisions and a
library call would hide them.

TWO PERIODS, deliberately:
  * 168 hours (a week) - the dominant cycle. Commuting, nightlife and weekends all live
    here, and it is the period the allocation actually plans against.
  * 24 hours (a day) - nested inside the weekly cycle. Reported separately so the weekly
    component can be read as "which day" rather than being confounded with "which hour".
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class Decomposition:
    period: int
    observed: List[float]
    trend: List[Optional[float]]
    seasonal: List[float]
    residual: List[Optional[float]]

    def strength(self) -> Dict[str, float]:
        """Seasonal and trend strength, as defined in Hyndman & Athanasopoulos.

        strength = max(0, 1 - Var(residual) / Var(residual + component))

        It answers "how much of what the trend cannot explain is explained by
        seasonality" on a 0-1 scale, which is directly comparable across series with
        completely different magnitudes - unlike an R-squared on the raw values.
        """
        paired = [(r, s) for r, s in zip(self.residual, self.seasonal) if r is not None]
        if len(paired) < 3:
            return {"seasonal_strength": 0.0, "trend_strength": 0.0}

        residuals = [r for r, _ in paired]
        seasonal_plus = [r + s for r, s in paired]
        detrended = [o - t for o, t in zip(self.observed, self.trend) if t is not None]

        var_r = statistics.pvariance(residuals)
        var_rs = statistics.pvariance(seasonal_plus)
        var_d = statistics.pvariance(detrended) if len(detrended) > 2 else 0.0

        return {
            "seasonal_strength": round(max(0.0, 1 - var_r / var_rs), 4) if var_rs else 0.0,
            "trend_strength": round(max(0.0, 1 - var_r / var_d), 4) if var_d else 0.0,
            "residual_sd": round(math.sqrt(var_r), 4),
        }


def moving_average(series: Sequence[float], window: int) -> List[Optional[float]]:
    """Centred moving average. Returns None where the window does not fit.

    None rather than a padded or extrapolated value: inventing endpoints makes the trend
    look confident exactly where there is least information, and the seasonal averages
    below would then be computed partly from fabricated numbers.

    An even window is averaged twice (a 2xM average), the standard treatment, because a
    centred average of an even number of points sits between observations rather than on
    one.
    """
    n = len(series)
    if window > n:
        return [None] * n

    if window % 2 == 0:
        half = window // 2
        out: List[Optional[float]] = [None] * n
        for i in range(half, n - half):
            window_slice = series[i - half:i + half + 1]
            # Endpoints get half weight - that is what makes it a 2xM average.
            weighted = (0.5 * window_slice[0] + sum(window_slice[1:-1])
                        + 0.5 * window_slice[-1])
            out[i] = weighted / window
        return out

    half = window // 2
    return [None] * half + [
        sum(series[i - half:i + half + 1]) / window for i in range(half, n - half)
    ] + [None] * half


def decompose(series: Sequence[float], period: int) -> Decomposition:
    series = list(series)
    n = len(series)
    if n < 2 * period:
        raise ValueError(f"need at least two full periods ({2 * period}), got {n}")

    trend = moving_average(series, period)

    # Seasonal component: average the detrended series within each position in the cycle,
    # then centre so the components sum to zero. Without centring the seasonal absorbs
    # part of the level and the trend is biased.
    by_position: Dict[int, List[float]] = {}
    for i, (observed, t) in enumerate(zip(series, trend)):
        if t is None:
            continue
        by_position.setdefault(i % period, []).append(observed - t)

    means = {pos: statistics.fmean(vals) for pos, vals in by_position.items()}
    if means:
        overall = statistics.fmean(means.values())
        means = {pos: value - overall for pos, value in means.items()}

    seasonal = [means.get(i % period, 0.0) for i in range(n)]
    residual = [None if t is None else observed - t - s
                for observed, t, s in zip(series, trend, seasonal)]

    return Decomposition(period, series, trend, seasonal, residual)


def hour_of_week_profile(series: Sequence[float]) -> Dict[str, object]:
    """The 168-hour cycle, plus the peak and trough it implies."""
    result = decompose(series, 168)
    seasonal = result.seasonal[:168]
    peak = max(range(168), key=lambda i: seasonal[i])
    trough = min(range(168), key=lambda i: seasonal[i])
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return {
        "period": 168,
        **result.strength(),
        "peak_hour_of_week": peak,
        "peak_label": f"{days[peak // 24]} {peak % 24:02d}:00",
        "peak_effect": round(seasonal[peak], 3),
        "trough_hour_of_week": trough,
        "trough_label": f"{days[trough // 24]} {trough % 24:02d}:00",
        "trough_effect": round(seasonal[trough], 3),
        "peak_to_trough": round(seasonal[peak] - seasonal[trough], 3),
        "seasonal_component": [round(v, 4) for v in seasonal],
    }
