"""
main.py

Purpose
-------
FastAPI backend for the CoolLoad AI dashboard. Serves two things:

1. Numeric optimizer results (KPIs, daily peak kW, per-day hourly
   dispatch including setpoint and indoor temperature) for the charts.
2. The RAG chat agent (agent.py) for the manager Q&A panel.

Also serves the dashboard's static index.html at "/", so the whole
demo runs from a single `uvicorn main:app` process with no separate
static file server needed.

Design decisions
-----------------
1. optimize_month() runs ONCE, either at startup (if no cache exists)
   or is loaded from a small on-disk cache (data/cache/). This avoids
   re-running the optimizer (which can take anywhere from ~1 minute
   to over an hour depending on enable_continuous_search) on every
   server restart during development.

   IMPORTANT CAVEAT, stated explicitly rather than silently assumed:
   this FastAPI process runs its OWN HVACOptimizer instance, separate
   from whatever `python optimizer.py` run last populated data/rag/
   for the chat agent. That means the chart numbers (from this file's
   own optimize_month() call) and the chat answers (from whatever
   optimizer.py run last wrote data/rag/) are only guaranteed to match
   if both were run with the same settings against the same data/model.
   This mirrors the exact mismatch class of bug fixed earlier in this
   project (export_rag_data.py re-running the optimizer with different
   settings than the console run) -- it hasn't reappeared here, it's
   just unavoidable with two separate processes reading/writing
   different artifacts. If exact consistency between charts and chat
   ever matters for a specific demo, run `python optimizer.py` first
   (which now also writes data/rag/), delete data/cache/, then start
   this server so both come from the same run.

   ENABLE_CONTINUOUS_SEARCH below defaults to False specifically so a
   fresh server start (no cache yet) finishes in roughly a minute
   instead of over an hour. Flip it to True (or set the
   COOLLOAD_FULL_SEARCH=1 environment variable) if you want the
   server's own numbers to reflect the full DE/Powell search.

2. Per-day indoor temperature (needed for the Setpoint vs Indoor
   Temperature chart) is NOT present in optimize_month()'s
   hourly_schedule output -- only setpoints and predicted loads are.
   Rather than modify optimizer.py again to add it, this file
   recomputes it independently using thermal_model.build_thermal_
   features() on the setpoint columns optimize_month() already
   produced. This is read-only with respect to optimizer.py: it does
   not change optimizer.py's code or its own output in any way.

3. RagAgent (agent.py) is instantiated once at startup, not per
   request -- it loads the Groq client and TF-IDF index once, then
   ask_with_sources() is called per chat message.

Setup
-----
    pip install fastapi "uvicorn[standard]"

Run (from the scripts/ directory, so the local imports below resolve
the same way every other script in this project already does):
    cd scripts
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 -- the dashboard is served from the
same process as the API.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from optimizer import (  # noqa: E402
    HVACOptimizer,
    DATA_PATH,
    PROJECT_ROOT,
    export_rag_documents,
)
from thermal_model import build_thermal_features  # noqa: E402
from agent import RagAgent  # noqa: E402  (local filename: agent.py)

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
HOURLY_CACHE_PATH = CACHE_DIR / "hourly_schedule.csv"
DAILY_CACHE_PATH = CACHE_DIR / "daily_strategies.csv"
SUMMARY_CACHE_PATH = CACHE_DIR / "summary.json"

# See design decision 1 above: False keeps a cold server start to
# roughly a minute. Set COOLLOAD_FULL_SEARCH=1 in the environment to
# run the full DE/Powell search instead (can take over an hour).
ENABLE_CONTINUOUS_SEARCH = os.getenv("COOLLOAD_FULL_SEARCH", "0") == "1"

# In-memory state populated once at startup, read by every request.
STATE: dict[str, Any] = {}


RAG_SUMMARY_JSON_PATH = PROJECT_ROOT / "data" / "rag" / "monthly_summary.json"


def _load_or_compute_results() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load cached optimizer output if present, otherwise compute it once.

    Returns (hourly_df, daily_df, summary_dict), matching the shape of
    optimize_month()'s "hourly_schedule", "daily_strategies", and
    "summary" fields.

    Consistency with data/rag/ (used by the chat agent): whichever
    branch below produces hourly_df/daily_df/summary, export_rag_
    documents() is called on those SAME objects ONLY IF data/rag/
    doesn't already contain a monthly_summary.json. This is
    deliberately NOT unconditional -- writing data/rag/ on every
    startup was correct for consistency but made every restart (even
    a cache-hit one) redo ~30 file writes for no reason, and worse,
    gave the impression a full re-optimization was happening. The
    real cost is the OPTIMIZER run, not the RAG export -- so the rule
    that actually matches "fast on repeat startups, but never drifts"
    is: regenerate data/rag/ only when it's missing (first run ever,
    or after someone deletes data/cache/ and data/rag/ together), and
    otherwise trust what's already on disk untouched.
    """
    if (
        HOURLY_CACHE_PATH.exists()
        and DAILY_CACHE_PATH.exists()
        and SUMMARY_CACHE_PATH.exists()
    ):
        print(f"[startup] Loading cached results from {CACHE_DIR}")
        hourly_df = pd.read_csv(HOURLY_CACHE_PATH)
        daily_df = pd.read_csv(DAILY_CACHE_PATH)
        summary = json.loads(SUMMARY_CACHE_PATH.read_text(encoding="utf-8"))
    else:
        print(
            "[startup] No cache found. Running optimizer once "
            f"(enable_continuous_search={ENABLE_CONTINUOUS_SEARCH})..."
        )
        df = pd.read_csv(DATA_PATH)
        optimizer = HVACOptimizer(enable_continuous_search=ENABLE_CONTINUOUS_SEARCH)
        results = optimizer.optimize_month(df)

        hourly_df = results["hourly_schedule"]
        daily_df = results["daily_strategies"]
        summary = results["summary"]

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        hourly_df.to_csv(HOURLY_CACHE_PATH, index=False)
        daily_df.to_csv(DAILY_CACHE_PATH, index=False)
        SUMMARY_CACHE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[startup] Cached results to {CACHE_DIR}")

    if not RAG_SUMMARY_JSON_PATH.exists():
        print(
            "[startup] data/rag/ not found -- writing it now from these "
            "SAME results, so charts and chat start out in sync..."
        )
        export_rag_documents(
            {
                "summary": summary,
                "daily_strategies": daily_df,
                "hourly_schedule": hourly_df,
            }
        )
    else:
        print("[startup] data/rag/ already present -- reusing it as-is (fast path).")

    return hourly_df, daily_df, summary


def _build_daily_peak(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> list[dict]:
    """One row per day: baseline peak kW (computed from hourly_df) vs.
    optimized peak kW (already computed by optimize_month() as
    daily_peak_kw -- the peak of whichever schedule was selected).
    """
    hourly_df = hourly_df.copy()
    hourly_df["date"] = pd.to_datetime(hourly_df["timestamp"]).dt.strftime("%Y-%m-%d")
    baseline_peak_by_date = hourly_df.groupby("date")["baseline_load_kw"].max()

    rows: list[dict] = []
    for _, row in daily_df.sort_values("date").iterrows():
        date = str(row["date"])
        rows.append(
            {
                "date": date,
                "baseline_peak_kw": round(
                    float(baseline_peak_by_date.get(date, 0.0)), 2
                ),
                "optimized_peak_kw": round(float(row["daily_peak_kw"]), 2),
            }
        )
    return rows


def _build_hourly_index(hourly_df: pd.DataFrame) -> dict[str, list[dict]]:
    """Per-date list of hourly rows, with indoor temperature added.

    Indoor temperature is recomputed here (not read from
    optimize_month()'s output, which doesn't include it) using
    thermal_model.build_thermal_features() against the setpoint
    columns optimize_month() already produced. Run once for the
    baseline setpoint series and once for the optimized setpoint
    series, per day.
    """
    hourly_df = hourly_df.copy()
    hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])
    hourly_df["date"] = hourly_df["timestamp"].dt.strftime("%Y-%m-%d")

    index: dict[str, list[dict]] = {}

    for date, day_df in hourly_df.groupby("date", sort=True):
        day_df = day_df.sort_values("hour").reset_index(drop=True)

        baseline_features = build_thermal_features(
            day_df, day_df["baseline_setpoint_c"].to_numpy()
        )
        optimized_features = build_thermal_features(
            day_df, day_df["optimized_setpoint_c"].to_numpy()
        )

        hours = []
        for i in range(len(day_df)):
            hours.append(
                {
                    "hour": int(day_df.loc[i, "hour"]),
                    "temperature_c": round(float(day_df.loc[i, "temperature_c"]), 2),
                    "occupancy_factor": round(
                        float(day_df.loc[i, "occupancy_factor"]), 2
                    ),
                    "baseline_setpoint_c": round(
                        float(day_df.loc[i, "baseline_setpoint_c"]), 2
                    ),
                    "optimized_setpoint_c": round(
                        float(day_df.loc[i, "optimized_setpoint_c"]), 2
                    ),
                    "baseline_load_kw": round(
                        float(day_df.loc[i, "baseline_load_kw"]), 2
                    ),
                    "optimized_load_kw": round(
                        float(day_df.loc[i, "optimized_load_kw"]), 2
                    ),
                    "baseline_indoor_temp_c": round(
                        float(baseline_features.loc[i, "indoor_temp_c"]), 2
                    ),
                    "optimized_indoor_temp_c": round(
                        float(optimized_features.loc[i, "indoor_temp_c"]), 2
                    ),
                }
            )
        index[str(date)] = hours

    return index


@asynccontextmanager
async def lifespan(app: FastAPI):
    hourly_df, daily_df, summary = _load_or_compute_results()

    STATE["summary"] = summary
    STATE["daily_peak"] = _build_daily_peak(hourly_df, daily_df)
    STATE["hourly_by_date"] = _build_hourly_index(hourly_df)
    STATE["dates"] = sorted(STATE["hourly_by_date"].keys())

    print("[startup] Loading RAG agent (requires data/rag/ and GROQ_API_KEY)...")
    try:
        STATE["rag_agent"] = RagAgent()
        print("[startup] RAG agent ready.")
    except Exception as exc:
        # Don't crash the whole API if the chat agent can't load (e.g.
        # missing GROQ_API_KEY or empty data/rag/) -- the chart
        # endpoints should still work for the demo even if chat is down.
        print(f"[startup] WARNING: RAG agent failed to load: {exc}")
        STATE["rag_agent"] = None

    yield
    STATE.clear()


app = FastAPI(title="CoolLoad AI API", lifespan=lifespan)

# Permissive CORS for local hackathon development. Tighten this
# (specific origins) before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "rag_agent_loaded": STATE.get("rag_agent") is not None,
        "days_loaded": len(STATE.get("dates", [])),
    }


@app.get("/api/summary")
def get_summary() -> dict:
    return STATE["summary"]


@app.get("/api/daily-peak")
def get_daily_peak() -> list[dict]:
    return STATE["daily_peak"]


@app.get("/api/dates")
def get_dates() -> list[str]:
    return STATE["dates"]


@app.get("/api/hourly/{date}")
def get_hourly(date: str) -> list[dict]:
    hourly_by_date: dict[str, list[dict]] = STATE["hourly_by_date"]
    if date not in hourly_by_date:
        raise HTTPException(status_code=404, detail=f"No data for date {date}")
    return hourly_by_date[date]


@app.post("/api/chat")
def post_chat(payload: ChatRequest) -> dict:
    agent: RagAgent | None = STATE.get("rag_agent")

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG agent is not available. Check GROQ_API_KEY in .env and "
                "that data/rag/ has been generated (run optimizer.py)."
            ),
        )

    try:
        print(f"[CHAT] Message: {payload.message}")
        result = agent.ask_with_sources(payload.message)
        print(f"[CHAT] Result: {result}")
        return result

    except Exception as exc:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=502,
            detail=f"{type(exc).__name__}: {exc}",
        )

# Mounted LAST and deliberately, so it never shadows the /api/* routes
# above -- Starlette matches routes in registration order, and this is
# a catch-all for "/".
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
