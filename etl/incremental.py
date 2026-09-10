"""Incremental load over partitioned Parquet, with a watermark.

THE PROBLEM THIS SOLVES. Twelve months is ~40 million rows and ~650 MB of Parquet.
Reloading everything on each run takes minutes and, more importantly, makes the pipeline
untestable - you cannot iterate on the transform if every iteration costs a full reload.

THE DESIGN.
  * **Partitioned output.** Each month is written to its own Hive-style partition
    directory (`month=2024-01/`). DuckDB and Spark both prune partitions from the path,
    so a query restricted to Q1 reads three files rather than twelve.
  * **A watermark table.** Each successfully loaded partition records its source file, its
    row counts and a content hash. A re-run skips partitions whose source is unchanged.
  * **Idempotent by construction.** Loading the same month twice produces the same
    partition, not duplicate rows. The watermark is checked BEFORE the write and the
    partition is replaced atomically, so an interrupted run leaves the previous partition
    intact rather than a half-written one.

WHY A CONTENT HASH AND NOT A TIMESTAMP. mtime changes when a file is copied, touched, or
restored from a backup - all of which would trigger a needless reload - and does NOT change
if a file is overwritten in place with the same mtime. Size plus a hash of the head and
tail is cheap and actually tracks content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("URBAN_DATA", ROOT / "data" / "raw"))
WAREHOUSE = Path(os.environ.get("URBAN_WAREHOUSE", ROOT / "data" / "warehouse"))
DB_PATH = ROOT / "data" / "urban.duckdb"

FILE_RE = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet$")


@dataclass
class PartitionResult:
    month: str
    source_file: str
    source_hash: str
    source_rows: int
    loaded_rows: int
    excluded_rows: int
    seconds: float
    skipped: bool


def content_hash(path: Path, sample_bytes: int = 1 << 20) -> str:
    """Size plus a hash of the first and last MB.

    Hashing 60 MB per file on every run would cost more than the reload it is meant to
    avoid. Head+tail+size catches truncation, replacement and corruption - the failure
    modes that actually occur here - without reading the middle of the file.
    """
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(sample_bytes))
        if size > 2 * sample_bytes:
            fh.seek(-sample_bytes, os.SEEK_END)
            h.update(fh.read(sample_bytes))
    return h.hexdigest()[:16]


def discover(raw: Path = RAW) -> List[Path]:
    return sorted(p for p in raw.glob("yellow_tripdata_*.parquet") if FILE_RE.search(p.name))


def month_of(path: Path) -> str:
    match = FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse a month from {path.name}")
    return f"{match.group(1)}-{match.group(2)}"


def ensure_watermark(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS load_watermark (
            month         VARCHAR PRIMARY KEY,
            source_file   VARCHAR NOT NULL,
            source_hash   VARCHAR NOT NULL,
            source_rows   BIGINT  NOT NULL,
            loaded_rows   BIGINT  NOT NULL,
            excluded_rows BIGINT  NOT NULL,
            loaded_at     TIMESTAMP NOT NULL
        )
    """)


def load_partition(conn: duckdb.DuckDBPyConnection, path: Path,
                   force: bool = False) -> PartitionResult:
    from .quality import build_case_expression

    month = month_of(path)
    digest = content_hash(path)
    started = time.perf_counter()

    existing = conn.execute(
        "SELECT source_hash, source_rows, loaded_rows, excluded_rows "
        "FROM load_watermark WHERE month = ?", (month,)).fetchone()
    if existing and existing[0] == digest and not force:
        return PartitionResult(month, path.name, digest, int(existing[1]),
                               int(existing[2]), int(existing[3]), 0.0, True)

    source_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]

    case_expr = build_case_expression(f"{month}-01")
    partition_dir = WAREHOUSE / f"month={month}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    target = partition_dir / "trips.parquet"
    tmp = partition_dir / "trips.parquet.tmp"

    # Project only the columns the analysis uses. The raw file has 19; carrying the
    # unused ten through a 40-million-row pipeline is pure I/O for nothing.
    conn.execute(f"""
        COPY (
            SELECT
                tpep_pickup_datetime                              AS pickup_ts,
                tpep_dropoff_datetime                             AS dropoff_ts,
                CAST(PULocationID AS INTEGER)                     AS pickup_zone_id,
                CAST(DOLocationID AS INTEGER)                     AS dropoff_zone_id,
                CAST(passenger_count AS INTEGER)                  AS passengers,
                trip_distance                                     AS distance_mi,
                total_amount                                      AS fare_total,
                fare_amount                                       AS fare_base,
                tip_amount                                        AS tip,
                DATE_DIFF('second', tpep_pickup_datetime, tpep_dropoff_datetime)
                                                                  AS duration_sec,
                HOUR(tpep_pickup_datetime)                        AS pickup_hour,
                DAYOFWEEK(tpep_pickup_datetime)                   AS pickup_dow,
                CAST(tpep_pickup_datetime AS DATE)                AS pickup_date,
                '{month}'                                         AS month
            FROM read_parquet('{path.as_posix()}')
            WHERE ({case_expr}) = 'ok'
        ) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    loaded_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{tmp.as_posix()}')").fetchone()[0]
    # Rename last: an interrupted run leaves the previous partition intact rather than a
    # truncated replacement.
    tmp.replace(target)

    conn.execute("""
        INSERT OR REPLACE INTO load_watermark
        VALUES (?, ?, ?, ?, ?, ?, now())
    """, (month, path.name, digest, source_rows, loaded_rows, source_rows - loaded_rows))

    return PartitionResult(month, path.name, digest, source_rows, loaded_rows,
                           source_rows - loaded_rows,
                           round(time.perf_counter() - started, 2), False)


def quality_breakdown(conn: duckdb.DuckDBPyConnection, path: Path) -> List[Dict[str, object]]:
    """Why rows were excluded, for one month. Run on demand - it is a second pass."""
    from .quality import build_case_expression

    month = month_of(path)
    case_expr = build_case_expression(f"{month}-01")
    rows = conn.execute(f"""
        SELECT {case_expr} AS status, COUNT(*) AS n
        FROM read_parquet('{path.as_posix()}')
        GROUP BY 1 ORDER BY n DESC
    """).fetchall()
    total = sum(n for _, n in rows)
    return [{"status": s, "rows": int(n), "pct": round(100.0 * n / total, 4)}
            for s, n in rows]


def run(force: bool = False, months: Optional[int] = None) -> Dict[str, object]:
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    ensure_watermark(conn)

    files = discover()
    if months:
        files = files[:months]
    if not files:
        raise FileNotFoundError(f"no parquet in {RAW}; run python3 data/fetch.py")

    results: List[PartitionResult] = []
    for path in files:
        result = load_partition(conn, path, force=force)
        results.append(result)
        note = "skipped (unchanged)" if result.skipped else f"{result.seconds:.1f}s"
        print(f"  {result.month}  {result.source_rows:>9,} -> {result.loaded_rows:>9,} "
              f"kept  ({result.excluded_rows:>7,} excluded, "
              f"{100 * result.excluded_rows / max(1, result.source_rows):>5.2f}%)  {note}",
              flush=True)

    conn.close()
    total_source = sum(r.source_rows for r in results)
    total_loaded = sum(r.loaded_rows for r in results)
    return {
        "partitions": [asdict(r) for r in results],
        "months": len(results),
        "source_rows": total_source,
        "loaded_rows": total_loaded,
        "excluded_rows": total_source - total_loaded,
        "excluded_pct": round(100.0 * (total_source - total_loaded) / max(1, total_source), 4),
        "warehouse": str(WAREHOUSE),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reload even if unchanged")
    ap.add_argument("--months", type=int, default=None)
    args = ap.parse_args()

    summary = run(force=args.force, months=args.months)
    print(f"\n  {summary['months']} partitions, {summary['loaded_rows']:,} rows kept, "
          f"{summary['excluded_pct']}% excluded")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "etl.json").write_text(json.dumps(summary, indent=2) + "\n")
