"""Assert the browser allocation matches the Python allocation exactly.

The static demo reimplements the bounded-heap top-k and the greedy capacity fill in
JavaScript so it runs with no backend. That is only honest if it agrees with the Python,
so this runs both over the real cell table and compares assignments and revenue.

Run:  python3 tools/check_js_matches_python.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocate import Cell, allocate_by_volume, allocate_hour, allocate_uniform  # noqa: E402

CASES = [(100, 18, 4), (300, 8, 1), (500, 2, 6), (1000, 12, 3), (50, 22, 5)]
POLICIES = {"revenue_per_hour": allocate_hour, "trip_volume": allocate_by_volume,
            "uniform": allocate_uniform}


def main() -> int:
    payload = json.loads((ROOT / "docs" / "cells.json").read_text())
    zones = payload["zones"]
    cells = [Cell(zone_id=z, zone_name=zones[str(z)][0], hour=h, dow=d,
                  expected_trips_per_hour=t, revenue_per_driver_hour=r,
                  borough=zones[str(z)][1])
             for z, d, h, t, r in payload["cells"]]

    script = f"""
const fs = require('fs');
const {{allocate}} = require('{ROOT / "docs" / "allocate.js"}');
const p = JSON.parse(fs.readFileSync('{ROOT / "docs" / "cells.json"}', 'utf8'));
const cases = {json.dumps(CASES)};
const out = {{}};
for(const [fleet, hour, dow] of cases)
  for(const policy of ['revenue_per_hour','trip_volume','uniform'])
    out[`${{fleet}}|${{hour}}|${{dow}}|${{policy}}`] = allocate(p.cells, fleet, hour, dow, policy);
console.log(JSON.stringify(out));
"""
    tmp = ROOT / "tools" / "_check.js"
    tmp.write_text(script)
    proc = subprocess.run(["node", str(tmp)], capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        print(proc.stderr[:900], file=sys.stderr)
        return 1
    js = json.loads(proc.stdout)

    ok = True
    print(f"  {'case':<26} {'python $':>13} {'js $':>13}  {'revenue':>8} {'cells':>6} {'placed':>7}")
    for fleet, hour, dow in CASES:
        for name, fn in POLICIES.items():
            py = fn(cells, fleet, hour, dow)
            j = js[f"{fleet}|{hour}|{dow}|{name}"]
            same_rev = abs(py.expected_revenue - j["expected_revenue"]) < 0.01
            same_cells = py.cells_used == j["cells_used"]
            same_placed = py.drivers_placed == j["drivers_placed"]
            ok &= same_rev and same_cells and same_placed
            print(f"  {f'{fleet}/{hour}h/d{dow} {name[:9]}':<26} "
                  f"{py.expected_revenue:>13,.0f} {j['expected_revenue']:>13,.0f}  "
                  f"{str(same_rev):>8} {str(same_cells):>6} {str(same_placed):>7}")

    print("\n  JavaScript agrees with Python exactly" if ok
          else "\n  *** THE TWO IMPLEMENTATIONS DIVERGE ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
