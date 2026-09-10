"""Serve the demand heatmap and the allocation, read-only over the warehouse."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocate import Cell, allocate_by_volume, allocate_hour, allocate_uniform  # noqa: E402

DB = Path(os.environ.get("URBAN_DB", ROOT / "data" / "urban.duckdb"))
OUTPUTS = ROOT / "outputs"
UI = ROOT / "dashboard" / "index.html"

app = FastAPI(title="Urban Demand Warehouse", version="1.0.0")
_cells_cache: Optional[List[Cell]] = None


def _conn():
    if not DB.exists():
        raise HTTPException(503, "warehouse not built; run make build")
    return duckdb.connect(str(DB), read_only=True)


def _cells() -> List[Cell]:
    """Cell profiles, loaded once. ~39,000 rows is a few MB and the allocation is called
    on every slider move, so re-querying per request would make the UI feel broken."""
    global _cells_cache
    if _cells_cache is None:
        conn = _conn()
        try:
            rows = conn.execute("""
                SELECT zone_id, ANY_VALUE(zone_name), ANY_VALUE(borough), dow, hour,
                       AVG(mean_trips_per_day), AVG(revenue_per_driver_hour)
                FROM mart_demand_profile
                WHERE revenue_per_driver_hour > 0
                GROUP BY zone_id, dow, hour""").fetchall()
        finally:
            conn.close()
        _cells_cache = [
            Cell(int(z), str(n), int(h), int(d), float(t or 0), float(r or 0), str(b))
            for z, n, b, d, h, t, r in rows]
    return _cells_cache


@app.get("/health")
async def health():
    return {"status": "ok" if DB.exists() else "no_warehouse",
            "warehouse": DB.exists(),
            "results": (OUTPUTS / "results.json").exists(),
            "cells": len(_cells()) if DB.exists() else 0}


@app.get("/api/summary")
async def summary():
    path = OUTPUTS / "results.json"
    if not path.exists():
        raise HTTPException(503, "no results; run make analyse")
    return json.loads(path.read_text())


@app.get("/api/heatmap")
async def heatmap(metric: str = Query("revenue_per_driver_hour"),
                  borough: Optional[str] = None):
    """The zone x hour-of-week grid the UI renders."""
    allowed = {"revenue_per_driver_hour", "mean_trips_per_day", "total_revenue"}
    if metric not in allowed:
        raise HTTPException(422, f"metric must be one of {sorted(allowed)}")

    conn = _conn()
    try:
        where = "WHERE revenue_per_driver_hour > 0"
        if borough:
            where += f" AND borough = '{borough.replace(chr(39), '')}'"
        rows = conn.execute(f"""
            SELECT hour_of_week, ANY_VALUE(dow) AS dow, ANY_VALUE(hour) AS hour,
                   SUM(total_trips) AS trips,
                   SUM(total_revenue) / NULLIF(SUM(total_driver_hours), 0) AS rph,
                   AVG(mean_trips_per_day) AS mean_trips
            FROM mart_demand_profile {where}
            GROUP BY hour_of_week ORDER BY hour_of_week""").fetchall()
        boroughs = [b for (b,) in conn.execute(
            "SELECT DISTINCT borough FROM dim_zone ORDER BY 1").fetchall()]
    finally:
        conn.close()

    return {"metric": metric, "borough": borough, "boroughs": boroughs,
            "cells": [{"hour_of_week": int(h), "dow": int(d), "hour": int(hr),
                       "trips": int(t or 0),
                       "revenue_per_driver_hour": round(float(r or 0), 2),
                       "mean_trips_per_day": round(float(mt or 0), 2)}
                      for h, d, hr, t, r, mt in rows]}


@app.get("/api/zones")
async def zones(order: str = "value", limit: int = 15):
    conn = _conn()
    try:
        column = ("revenue_per_driver_hour" if order == "value" else "trips")
        having = "WHERE trips > 20000" if order == "value" else ""
        rows = conn.execute(f"""
            SELECT zone_name, borough, trips, revenue, revenue_per_driver_hour
            FROM mart_zone_summary {having}
            ORDER BY {column} DESC LIMIT {int(limit)}""").fetchall()
    finally:
        conn.close()
    return {"order": order,
            "zones": [{"zone": z, "borough": b, "trips": int(t), "revenue": float(r),
                       "revenue_per_driver_hour": round(float(rph), 2)}
                      for z, b, t, r, rph in rows]}


@app.get("/api/allocate")
async def allocate(fleet: int = Query(300, ge=1, le=20000),
                   hour: int = Query(18, ge=0, le=23),
                   dow: int = Query(3, ge=0, le=6),
                   policy: str = Query("revenue_per_hour")):
    """Run the allocation live. This is the interactive part."""
    policies = {"revenue_per_hour": allocate_hour, "trip_volume": allocate_by_volume,
                "uniform": allocate_uniform}
    if policy not in policies:
        raise HTTPException(422, f"policy must be one of {sorted(policies)}")
    result = policies[policy](_cells(), fleet, hour, dow)
    body = result.as_dict()
    body["assignments"] = body["assignments"][:25]
    return body


@app.get("/api/compare")
async def compare(fleet: int = Query(300, ge=1, le=20000),
                  hour: int = Query(18, ge=0, le=23),
                  dow: int = Query(3, ge=0, le=6)):
    """All three policies on the same hour, so the difference is visible at a glance."""
    cells = _cells()
    out = {}
    for name, fn in (("revenue_per_hour", allocate_hour),
                     ("trip_volume", allocate_by_volume),
                     ("uniform", allocate_uniform)):
        r = fn(cells, fleet, hour, dow)
        out[name] = {"expected_revenue": round(r.expected_revenue, 2),
                     "revenue_per_driver": round(
                         r.expected_revenue / fleet, 2) if fleet else 0,
                     "cells_used": r.cells_used, "drivers_idle": r.drivers_idle,
                     "top_zones": [a.zone_name for a in r.assignments[:5]]}
    best = max(out.values(), key=lambda v: v["expected_revenue"])["expected_revenue"]
    for value in out.values():
        value["vs_best_pct"] = round(100 * (value["expected_revenue"] - best) / best, 2) \
            if best else 0.0
    return {"fleet": fleet, "hour": hour, "dow": dow, "policies": out}


@app.get("/api/backtest")
async def backtest():
    path = OUTPUTS / "backtest.json"
    if not path.exists():
        raise HTTPException(503, "no backtest; run make backtest")
    return json.loads(path.read_text())


@app.get("/", response_class=HTMLResponse)
async def index():
    return UI.read_text() if UI.exists() else "<h1>Urban Demand</h1><p>UI not built.</p>"
