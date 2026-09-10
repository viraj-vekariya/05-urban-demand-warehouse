# Decisions — Urban Demand Warehouse

Every non-obvious choice, the alternatives, and why they lost.

---

## D-01 · Revenue per driver-hour is the allocation metric

**Chose:** rank cells by `SUM(revenue) / SUM(driver_hours)`.

**The alternatives, and why each is wrong:**
- *Total revenue* rewards cells that are merely busy. Midtown at lunch has enormous total
  revenue spread across enormous supply.
- *Trip count* rewards short cheap trips. It is the metric most dashboards implicitly
  encourage — "send drivers where the demand is" — and it is measurably worse.
- *Mean fare* ignores how long the fare takes to earn.

Revenue per driver-hour is what a driver actually optimises: a $70 airport run occupying
50 minutes beats four $12 crosstown hops occupying the same 50 minutes plus the
deadheading between them.

**Measured, not assumed:** the backtest ranks by trip volume as a real baseline. It loses
by a mean of 15.98% on held-out months. Had it not, the whole metric argument would be
decoration.

**And computed from SUMS, never as an average of ratios.** Averaging per-month ratios
weights a month with 3 trips the same as one with 30,000, which makes tiny cells look
enormously profitable and sends the fleet to noise. A test asserts the profile divides
totals rather than averaging ratios.

## D-02 · A bounded min-heap for top-k

**Chose:** a heap of size k holding the best-so-far, worst at the root.

**Why not just sort:** sorting all cells is O(n log n) and produces a full ordering nobody
reads. The heap is O(n log k) with **O(k) memory**, and a candidate worse than the current
worst is discarded in O(1) without ever entering the heap.

**The honest caveat:** at 39,033 cells the wall-clock difference is small. The reason it is
still correct is that the candidate set scales with zones × granularity and the fleet does
not — at per-block granularity there are millions of cells and still 500 drivers.

**A monotonic tie-breaker is inserted into the heap key.** Without it, equal revenues make
Python compare the `Cell` objects and raise `TypeError`. Real data has ties, and a test
constructs 50 identical cells.

**Verified against a full sort**, because the justification is that it is cheaper, not
different.

## D-03 · Capacity derived from the cell's own demand

**Chose:** `capacity = observed_trips_per_hour × 1.5`.

**Why any capacity at all:** naive top-k sends every driver to the single best cell. A cell
that historically served 40 trips an hour cannot absorb 200 drivers; the 201st earns
nothing. The constraint is what turns a sort into a greedy fill.

**Why 1.5× rather than 1.0:** the forecast is not exact, and a cell that averaged 40 trips
can plausibly see 60. Headroom, but bounded — the multiple is a choice and is stated as
one, not presented as a measurement.

**Why greedy is provably optimal here** (worth stating, because greedy usually is not):
every driver-hour is identical and independent, and capacities are hard constraints. Taking
cells in descending value until the fleet is exhausted cannot be beaten.

## D-04 · Solve each hour independently, and say what that costs

**Chose:** 24 independent single-hour allocations per day.

**What it ignores:** a driver who ends an hour at JFK begins the next hour at JFK.
Repositioning cost is not modelled.

**What the correct version would be:** a min-cost flow over a time-expanded network, with
arcs for both serving and repositioning. That is a substantially larger problem and a
different project.

**Why this is acceptable:** the simplification is stated in the code, in the README's known
limits, and here — and the backtest measures what the simplified policy actually earns
rather than assuming it is close to optimal.

## D-05 · Partitioned Parquet with a content-hash watermark

**Chose:** one Hive-style partition per month, plus a watermark table recording each
partition's source hash and row counts.

**Why partitioned:** DuckDB and Spark both prune partitions from the path, so the
backtest's "train on Jan–Sep" reads nine files rather than twelve.

**Why a content hash rather than mtime:** mtime changes when a file is copied, touched or
restored from backup — all of which would trigger a needless reload — and does *not* change
if a file is overwritten in place with the same timestamp. Size plus a hash of the head and
tail is cheap and actually tracks content.

**Why not hash the whole file:** 60 MB per file per run would cost more than the reload it
avoids. Head + tail + size catches truncation, replacement and corruption, which are the
failure modes that actually occur.

**Atomic rename on write**, so an interrupted run leaves the previous partition intact
rather than a truncated replacement that reads as a valid but short file.

## D-06 · Classify rows in the quality gate; never delete silently

**Chose:** a single ordered `CASE` assigning exactly one status per row.

**Why one expression rather than nine passes:** the WHEN order guarantees the statuses
*partition* the data. Nine separate boolean columns would let a row be counted in several
buckets, and the reconciliation would then fail for a reason unrelated to data quality.

**Why every rule carries a written reason:** a quality rule without a stated justification
is a rule nobody can challenge — and quality rules silently bias analyses. A test asserts
every rule has one.

## D-07 · A missing `passenger_count` does NOT disqualify a trip

**This is the most consequential decision in the project, and it was originally wrong.**

The first version excluded NULL and zero passenger counts along with implausible ones. That
single rule accounted for **12.66 of a 17.97% exclusion rate** — 483,731 trips in one month,
of which **89.5% were otherwise completely valid**.

And they were not a random slice: **mean distance 20.11 miles** against a fleet average near
3, i.e. disproportionately airport runs. The rule was removing an eighth of all demand *and
biasing it against exactly the high-value zones the allocation exists to find.*

`passenger_count` is used nowhere in the demand or revenue calculation. Fixing it took
exclusions from **14.71% to 4.78%** and recovered **4.1 million real trips**.

**The generalisable lesson:** a quality rule on a field the analysis does not use can only
cost you data. Every exclusion rule should have to justify itself against the specific
question being asked, and a high exclusion rate should be investigated rather than accepted
as evidence of rigour.

**Guarded by a CI gate** at 10%, because this is exactly the kind of thing a later refactor
reinstates without noticing.

## D-08 · Zones 264 and 265 are excluded

TLC's own data dictionary defines them as "Unknown" and "NA". They are not places.
Aggregated into the mart they become two enormous phantom hotspots — and the allocation,
which chases value, would send the fleet to them.

## D-09 · Generate the 168-hour spine; do not derive it

**Chose:** `dim_time` is a cross join of 7 days and 24 hours.

**Why:** deriving it from observed trips silently drops cells with zero demand — and
"nobody hails a cab in this zone at 4am" is precisely the information an allocation needs.
A spine built from the data can only ever describe the data.

## D-10 · `trips_per_day`, not raw trip totals

A month contains four or five of each weekday. Comparing raw totals across months ranks a
five-Monday month above a four-Monday one for no real reason. Normalising by
`days_observed` makes cells comparable across months of different lengths.

## D-11 · The forecast must beat the *cell mean*, not a global average

**Chose:** the baseline is each cell's own historical mean, with fallbacks to the zone mean
and the global mean for cells never seen in training.

**Why the fallbacks matter:** a baseline that skipped unseen cells would be scored on an
easier problem than the model. It has to face the same rows.

**The result:** the gradient-boosted model scores MAE 2.6763 against the baseline's 2.6699
— **0.24% worse**. It does not earn its place, and the pipeline reports that.

**Why this is the right way round:** demand here is overwhelmingly periodic (seasonal
strength 1.0000). A model that beat a global average would prove nothing; the arithmetic
baseline is the honest opponent, and it wins.

**Predictions are clipped at zero.** A tree ensemble can extrapolate below zero on sparse
cells, and a negative forecast would corrupt the capacity calculation downstream.

**Cyclical sin/cos encoding for hour and weekday**, so hour 23 sits next to hour 0 by
construction rather than the model having to spend splits learning it.

## D-12 · History features are computed from training months only

The single most common leak in time-series feature engineering. Computing `cell_history_mean`
over all months would put the test period inside the training features — and it would
inflate the *model* while leaving the *baseline* honest, which is exactly backwards.

## D-13 · Temporal backtest split, never random

**Chose:** train on Jan–Sep, test on Oct–Dec.

**Why:** a random split puts the same week on both sides, and the policy is then evaluated
on demand it was built from. Time series split by time.

**And revenue is realised against the TEST months' observed rates**, with a capacity
ceiling. Scoring a policy against the numbers it was built from measures nothing, and
without the ceiling an over-committing policy scores brilliantly on paper.

**Drivers sent to a cell that does not appear in the test months earn nothing.** That is the
correct treatment — the plan sent them somewhere with no demand — and it is what penalises
over-fitting to quiet training cells.

## D-14 · Three policies, one of which is the naive default

**Chose:** revenue-ranking against trip-volume ranking and a uniform spread.

**Why trip-volume is not a straw man:** it is what "send drivers where the demand is"
means, and it is the policy most demand dashboards implicitly encourage. A test asserts it
genuinely chases trip count, so the comparison is against the real alternative.

**Why uniform is included:** it is the "do nothing clever" floor. A policy that could not
beat spreading the fleet evenly would not be worth deploying.

## D-15 · PySpark implemented for real, and DuckDB used anyway

**Chose:** a genuine distributed implementation, cross-verified, and not on the serving
path.

**The Spark decisions that are actually Spark decisions:**
- **Broadcast the 263-row zone dimension explicitly.** Without the hint Spark may plan a
  sort-merge join and shuffle 39 million rows to meet a table that fits in a page. This is
  the most common avoidable cost in a Spark pipeline.
- **Aggregate before joining.** 39M rows reduce to ~39k cells first; joining first would
  carry zone-name strings through every row.
- **`spark.sql.shuffle.partitions = 8`, not the default 200.** With ~39,000 output rows,
  200 tasks each handle a couple of hundred records and scheduling overhead dwarfs the
  work.

**The honest result:** exact agreement with DuckDB on trips, cells, revenue and
driver-hours — and **DuckDB is 56× faster**. It builds the whole mart in 844 ms, less than
the JVM takes to start.

**So why keep it:** two independent implementations agreeing is a much stronger correctness
claim than either alone, and writing the aggregation against a distributed execution model
is a genuinely different exercise. It is not kept because it is faster, and claiming
otherwise at this scale would be theatre.

**A tolerance of $1 on a $1.1bn revenue sum** in the verification, because float addition
over 39 million rows genuinely differs in the last places depending on order, and the two
engines add in different orders.

## D-16 · Reconciliation failures abort the build

Nine checks, and `src/build.py` exits non-zero if any disagree. A mart that does not tie
back to its partitions is a mart nobody should build an allocation on, and a check that
merely warns is a check that gets ignored on the day it matters.

## D-17 · Seasonal decomposition before any model

**Chose:** classical additive decomposition on the 168-hour cycle, written out.

**Why before the model:** it answers how much of the variation is simple periodicity. At
**seasonal strength 1.0000**, a gradient-boosted model is an expensive way to learn "Friday
evening is busy" — and that is exactly what the forecast comparison then confirmed. The two
findings corroborate each other from different directions.

**Why `None` rather than padded endpoints:** inventing endpoint values makes the trend look
confident precisely where there is least information, and the seasonal averages would then
be computed partly from fabricated numbers.

**Seasonal components are centred to sum to zero**, or the seasonal absorbs part of the
level and the trend is biased.

## D-18 · The dashboard's interactive core is the fleet slider

**Chose:** the fleet size drives a live server-side allocation over all 39,033 cells.

**Why that control and not another:** the backtest's most interesting result is that the
advantage of revenue-ranking *decays* as the fleet grows — +17.8% at 100 drivers, +11.4% at
1,000 — because the high-value cells saturate. A slider makes that visible in a way a table
of four rows does not, and it is the result most likely to be misremembered as "clever
allocation always wins by 16%".
