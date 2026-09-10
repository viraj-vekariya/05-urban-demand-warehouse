# Urban Demand Warehouse

**Where should the fleet be tomorrow?** A partitioned Parquet warehouse over **39.2 million
real NYC taxi trips**, a SQL demand mart, a capacity-constrained allocation algorithm, and
a backtest on held-out months that says how much the allocation is actually worth.

**~3,400 lines · 48 tests passing · verified on CPython 3.13.9, PySpark 4.2.0, Temurin 21**

Every number was measured on a run and written to `outputs/`. Nothing is estimated.

---

## The scale

| | |
|---|---|
| source | NYC TLC yellow-cab trip records, 2024, public, no credentials |
| raw | 41.2M trips across 12 monthly Parquet files, 0.69 GB |
| loaded | **39,203,425 trips**, 4.78% excluded |
| revenue | **$1,103,993,060** |
| driver-hours | 11,031,247 |
| cells | **39,033** (zone × weekday × hour) |
| reconciliation | **9/9 checks pass**, and the build aborts if any fails |

---

## Finding 1 — a quality rule was quietly deleting an eighth of all demand

The first working pipeline excluded **14.71%** of trips. That is high enough to be worth
explaining, so I looked at the breakdown rather than accepting it:

```
implausible_passengers   12.66%   <- 70% of ALL exclusions
implausible_distance      1.68%
non_positive_fare         1.45%
implausible_duration      1.26%
unknown_zone              0.88%
```

The rule treated a **NULL** `passenger_count` as disqualifying. In one month that was
**483,731 trips** — and **89.5% of them were otherwise completely valid**: real fares, real
zones, real durations. Some TLC vendors simply stopped reporting the field.

Worse, they were not a random slice. Their **mean distance was 20.11 miles** against a
fleet average near 3 — disproportionately airport runs. Excluding them would have removed
an eighth of all demand *and systematically understated revenue in exactly the high-value
zones the allocation exists to find.*

`passenger_count` is not used to compute demand or revenue. Fixing the rule:

| | before | after |
|---|---:|---:|
| exclusion rate | 14.71% | **4.78%** |
| trips kept | 35,112,645 | **39,203,425** |

**4.1 million real trips recovered**, and the remaining exclusions are all genuinely bad
data. A CI gate now fails the build if the exclusion rate climbs back above 10%.

## Finding 2 — the allocation is worth +16%, and the advantage decays

Backtest: cell profiles built from **Jan–Sep**, revenue realised against **Oct–Dec's actual
observed rates**, with a capacity ceiling so drivers sent to a cell with no demand earn
nothing.

| fleet | revenue policy | trip-volume | uniform | lift vs best baseline |
|---:|---:|---:|---:|---:|
| 100 | $2,169,503 | $1,842,185 | $1,448,374 | **+17.77%** |
| 300 | $6,440,931 | $5,425,622 | $3,150,718 | **+18.71%** |
| 500 | $10,371,293 | $8,939,127 | $5,218,345 | **+16.02%** |
| 1000 | $19,158,828 | $17,194,139 | $8,018,815 | **+11.43%** |

**Mean lift +15.98%** over ranking by trip volume, and up to **+138.92%** over a uniform
spread.

**The lift decays as the fleet grows** — +17.8% at 100 drivers down to +11.4% at 1,000.
That is not noise, it is the mechanism: the high-value cells saturate, and every additional
driver must be sent somewhere progressively worse. Clever allocation is worth most to a
*small* fleet. The dashboard's fleet slider shows the same decay live.

## Finding 3 — the gradient-boosted model loses to arithmetic

The forecast was compared against the honest baseline — each cell's own historical mean —
rather than against a global average:

| model | MAE | RMSE | R² |
|---|---:|---:|---:|
| historical cell mean | **2.6699** | 8.2072 | 0.9810 |
| gradient-boosted trees | 2.6763 | 8.2561 | 0.9808 |

**The model is 0.24% *worse*.** It does not earn its place, and the pipeline says so in
`outputs/results.json`: *"the model does NOT beat the historical-mean baseline; use the
baseline."*

The seasonal decomposition explains why. **Seasonal strength = 1.0000** on the 168-hour
cycle — demand is essentially perfectly periodic, swinging 446,142 trips between the
Thursday 18:00 peak and the Tuesday 03:00 trough. There is almost nothing left for a model
to learn that "the same zone, same hour, same weekday" does not already capture.

This is the discipline a CV2 project is supposed to demonstrate: **the model is the
payload, never the point**, and it is allowed in only if it beats the arithmetic. Here it
does not, and reporting that is more useful than shipping it anyway.

## Finding 4 — busiest is not most valuable

| | zone | trips | $/driver-hour |
|---|---|---:|---:|
| busiest | Upper East Side South | 1,862,329 | $101.84 |
| most valuable | **LaGuardia Airport** | 1,233,244 | **$128.39** |

**The top 12 zones by volume and the top 12 by value share only 4 zones.**

This is why the allocation ranks by revenue per driver-hour rather than trip count. A
$70 airport run occupying 50 minutes beats four $12 crosstown hops occupying the same 50
minutes plus the deadheading between them. An allocation optimising trip count sends the
whole fleet to Midtown at lunchtime — and the backtest measures exactly what that costs.

---

## The algorithm

Ranking uses a **bounded min-heap**: O(n log k) time and **O(k) memory**, where k is the
fleet size, not the number of candidate cells. A test asserts it returns exactly what a
full sort would.

At 39,033 cells the wall-clock difference against sorting is small, and that is stated
honestly. The reason it is still the right structure is that the candidate set scales with
zones × granularity while the fleet does not — at per-block granularity there are millions
of cells and still 500 drivers.

**The capacity constraint is what makes it non-trivial.** Naive top-k sends every driver to
the single best cell; a cell that historically served 40 trips an hour cannot absorb 200
drivers, and the 201st earns nothing. Each cell's capacity is derived from its own observed
demand, which turns a sort into a greedy capacity-constrained fill — provably optimal here,
because every driver-hour is identical and independent and the capacities are hard.

**What it is not:** an optimal assignment. That would need deadheading, driver positions,
and the fact that a driver sent to JFK ends the hour at JFK — a min-cost flow over a
time-expanded network. The greedy fill is the honest 90% solution, and the backtest
measures what it earns rather than assuming it wins.

## PySpark, and why DuckDB is used anyway

`etl/spark_job.py` implements the same aggregation as a distributed job — partition
pruning, an explicit broadcast join for the 263-row zone dimension, aggregate-before-join,
and shuffle partitions tuned to the data instead of Spark's default 200.

Verified against DuckDB on the same months:

```
ok   trips          spark        9,101,490   duckdb        9,101,490
ok   cells          spark           32,060   duckdb           32,060
ok   revenue        spark   246,696,744.98   duckdb   246,696,744.98
ok   driver_hours   spark   2,350,332.3489   duckdb   2,350,332.3489
```

Exact agreement, to the cent. **And DuckDB was 56× faster** — it builds the entire demand
mart in 844 ms, less time than the JVM takes to start.

That is the honest result and the reason the pipeline uses DuckDB. Spark is here because
writing the aggregation against a distributed execution model is a genuinely different set
of decisions, and because two independent implementations agreeing is a far stronger
correctness claim than either alone. It is not here because it is faster, and pretending
otherwise at 39 million rows would be theatre.

---

## Layout

| path | lines | what |
|---|---:|---|
| `etl/` | ~560 | quality rules, watermarked incremental load, the PySpark job |
| `sql/` | 235 | zone/time dimensions, trip fact, demand mart, revenue marts, reconciliation |
| `src/` | ~1,000 | bounded-heap allocation, backtest, forecast + baseline, seasonal decomposition |
| `dashboard/` | ~410 | FastAPI + the live heatmap and allocation UI |
| `tests/` | ~490 | 48 tests |
| `data/` + `infra/` + CI | ~430 | fetcher, Dockerfile, compose, Fly, GitHub Actions |

## Run it

```bash
make setup
make all          # data → etl → build → backtest → analyse → test
make serve        # http://localhost:8500
```

**Drag the fleet slider.** The allocation re-runs server-side against 39,033 real cells and
you can watch the advantage of revenue-ranking narrow as the fleet grows — the same decay
the backtest measures on held-out months.

```bash
make data etl     # 12 months, watermarked and idempotent — a second run skips everything
make build        # 9/9 reconciliation checks
make backtest     # policies on Oct–Dec
make spark        # the distributed implementation, verified against DuckDB
make docker
```

## Known limits

- **Pickup-side only.** Demand is measured where trips *started*. A driver who ends an hour
  at JFK starts the next hour at JFK, and the hourly allocations are solved independently —
  so repositioning cost is not modelled at all.
- **No competition or elasticity.** The backtest assumes historical revenue rates hold when
  supply changes. Sending 500 drivers to a zone would in reality drive down each one's
  earnings, and this model has no way to represent that. It biases the measured lift
  **upward**, which is the direction that flatters the result — worth saying plainly.
- **Yellow cabs only.** Green cabs and for-hire vehicles are a large and growing share of
  NYC trips and are excluded entirely.
- **Capacity is a heuristic** — 1.5× historical trip rate. It is derived from data rather
  than assumed, but the multiple itself is a choice, not a measurement.
- **2024 only.** No year-over-year trend, and no ability to separate seasonality from
  secular change.
- **The Docker image builds 3 months, not 12**, to keep it a reasonable size. The pipeline
  is identical; `make all` locally reproduces the full year.

See `DECISIONS.md` for why each choice was made and what was rejected.
