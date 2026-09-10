"""Download real NYC TLC trip records, month by month.

Source: the NYC Taxi & Limousine Commission's public trip-record archive. No credentials,
no rate limit, ~50 MB of Parquet per month, roughly 3 million trips each. Twelve months is
about 40 million rows - large enough that the ETL decisions in etl/ are real decisions
rather than ceremony.

Two properties this file guarantees, both of which matter downstream:

* **Idempotent.** A month already on disk with a plausible size is not re-downloaded. The
  pipeline is run many times while developing and re-fetching 600 MB each time is both
  slow and rude to a public host.
* **Atomic.** Each file is written to a .part and renamed only on success. An interrupted
  download must not leave a truncated Parquet behind that reads as a valid but short file
  and silently drops a third of a month's trips.

Run:  python3 data/fetch.py [--months 12] [--year 2024]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("URBAN_DATA", ROOT / "data" / "raw"))
BASE = "https://d37ci6vzurychx.cloudfront.net"
UA = {"User-Agent": "Mozilla/5.0 (compatible; placement-project/1.0)"}

# A month of yellow-cab Parquet is ~45-55 MB. Anything much smaller is a truncated or
# error-page download masquerading as data.
MIN_PLAUSIBLE_BYTES = 5_000_000


def _download(url: str, dest: Path) -> Dict[str, object]:
    if dest.exists() and dest.stat().st_size > MIN_PLAUSIBLE_BYTES:
        return {"file": dest.name, "bytes": dest.stat().st_size, "cached": True}

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    started = time.time()
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=180) as response, part.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)

    size = part.stat().st_size
    if size < MIN_PLAUSIBLE_BYTES and dest.suffix == ".parquet":
        part.unlink()
        raise ValueError(f"{dest.name} downloaded only {size} bytes; refusing to keep it")

    # Rename last. On POSIX this is atomic, so the file either does not exist or is
    # complete - there is no state where a reader sees a half-written Parquet.
    part.replace(dest)
    return {"file": dest.name, "bytes": size, "cached": False,
            "seconds": round(time.time() - started, 1)}


def fetch_zones() -> Path:
    dest = RAW / "taxi_zone_lookup.csv"
    _download(f"{BASE}/misc/taxi_zone_lookup.csv", dest)
    return dest


def fetch_months(year: int = 2024, months: int = 12,
                 service: str = "yellow") -> List[Dict[str, object]]:
    results = []
    for month in range(1, months + 1):
        name = f"{service}_tripdata_{year}-{month:02d}.parquet"
        try:
            results.append(_download(f"{BASE}/trip-data/{name}", RAW / name))
            status = "cached" if results[-1]["cached"] else "downloaded"
            print(f"  {name}  {results[-1]['bytes'] / 1e6:>6.1f} MB  {status}", flush=True)
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as exc:
            # A missing month is survivable - TLC publishes with a lag, so the most
            # recent months of a year may not exist yet. It is recorded, not fatal.
            print(f"  {name}  SKIPPED ({exc})", file=sys.stderr)
            results.append({"file": name, "error": str(exc)[:120]})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--service", default="yellow")
    args = ap.parse_args()

    print(f"fetching {args.service} {args.year}, {args.months} months")
    zones = fetch_zones()
    files = fetch_months(args.year, args.months, args.service)

    ok = [f for f in files if "error" not in f]
    summary = {
        "year": args.year, "service": args.service,
        "zone_lookup": str(zones),
        "months_requested": args.months,
        "months_available": len(ok),
        "total_bytes": sum(f["bytes"] for f in ok),
        "files": files,
    }
    (RAW / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n  {len(ok)}/{args.months} months, "
          f"{summary['total_bytes'] / 1e9:.2f} GB in {RAW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
