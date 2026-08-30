# CoolLoad AI

AI-powered HVAC energy optimization for commercial buildings, built for the
**FortyGuard Temperature Hackathon**. Predicts hourly HVAC electrical load
for a target building, searches for a setpoint schedule that lowers energy
cost and monthly demand charges while respecting comfort limits, and lets a
building manager ask natural-language questions about the resulting plan.

Target building: Bank of America Tower, Phoenix, AZ (33.4479, -112.0704).

## Architecture

```
FortyGuard Weather API
        │  (heatmap + env_params, cached to disk)
        ▼
Building Profile
        │
        ▼
Thermal Model (scripts/thermal_model.py)
        │  first-order RC model: outdoor temp + setpoint → indoor temp → load
        ▼
Data Augmentation (scripts/augment_data.py)
        │  30 real days × ~60 synthetic setpoint schedules × 24h
        ▼
XGBoost Forecaster (scripts/train_model.py → models/hvac_forecaster.joblib)
        │  monotonic-constrained: load never rises with setpoint, never
        │  falls with outdoor/indoor temp
        ▼
Schedule Optimizer (scripts/optimizer.py — HVACOptimizer)
        │  discrete templates → Powell polish → differential evolution →
        │  monthly coordinate descent → demand-charge peak refinement
        ▼
Comfort Check + Savings Calculator (methods inside HVACOptimizer)
        │  hard comfort band + soft adaptive-comfort penalty;
        │  energy cost (TOU rate) + demand charge (peak kW) vs. baseline
        ▼
RAG AI Agent (scripts/agent.py + scripts/retrieve.py)
        │  TF-IDF retrieval over per-day + monthly docs → Groq (openai/gpt-oss-120b)
        ▼
FastAPI Backend (scripts/main.py)
        │  serves chart data (/api/*) and chat (/api/chat); also serves the dashboard
        ▼
Dashboard (dashboard/index.html)
           single-file Tailwind + ApexCharts + marked.js console
```

## Repository layout

```
fortyguard/                 FortyGuard tOS API client
  client.py                   class-based client — source of truth for the API contract
  exceptions.py               FortyGuardError, TaskFailedError, TaskTimeoutError, ActivityNotReadyError
  samples.py                   ready-made AOIs + filter-type constants

scripts/
  fetch.py                    pulls 30 days × 24h of real FortyGuard data → data/hvac_30day_dataset.csv
  thermal_model.py             RC thermal model + synthetic training-row generator
  augment_data.py              CLI wrapper around thermal_model.make_dynamic_training_rows
  train_model.py               trains models/hvac_forecaster.joblib
  optimizer.py                  HVACOptimizer: search, comfort check, savings calc, RAG export
  agent.py                     RagAgent: Groq-backed Q&A over exported RAG docs
  retrieve.py                  TF-IDF retriever + date-aware document selection
  main.py                      FastAPI app: chart endpoints, /api/chat, serves dashboard/

dashboard/index.html           single-file HTML/JS console

data/
  hvac_30day_dataset.csv        raw fetched weather + naive baseline load
  hvac_30day_dataset_augmented.csv   generated training rows
  cache/                       FastAPI startup cache (hourly_schedule.csv, daily_strategies.csv, summary.json)
  rag/                         exported RAG documents (days/*.txt+json, monthly_summary.txt+json)

models/hvac_forecaster.joblib  trained XGBoost forecaster
requirements.txt
.env                            FORTYGUARD_API_KEY, GROQ_API_KEY (not committed)
```

## Setup

```bash
conda create -n env3.10 python=3.10
conda activate env3.10
pip install -r requirements.txt
```

Create `.env` in the project root:
```
FORTYGUARD_API_KEY=your_fortyguard_key
GROQ_API_KEY=gsk_your_groq_key
```

## Running

Data, the trained model, and optimizer output are already generated and
cached in this repo, so day-to-day / demo use only needs:

```bash
cd scripts
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — dashboard and API are served from the same
process. On startup `main.py` loads `data/cache/` if present; otherwise it
runs the optimizer once (discrete-template search only by default, ~1
minute). Set `COOLLOAD_FULL_SEARCH=1` to run the full DE/Powell search on a
cold start instead (can take over an hour).

To regenerate everything from scratch:

```bash
python scripts/fetch.py          # pulls real weather data — uses FortyGuard API credits
python scripts/augment_data.py   # physics simulation, no API cost
python scripts/train_model.py
python scripts/optimizer.py      # console run; also exports data/rag/ for the chat agent
```

## Key design decisions

- **Single source of truth for computation.** `optimizer.py`'s RAG export
  runs on the exact same `hourly_df`/`daily_df`/`summary` objects the
  optimizer just produced, never a second re-run with different settings —
  this is what keeps dashboard charts and chat answers consistent. Caveat:
  `main.py` runs its own `HVACOptimizer` instance separate from whatever
  `python scripts/optimizer.py` last wrote to `data/rag/`; if exact
  consistency matters, run `optimizer.py` first, delete `data/cache/`, then
  start `main.py` so both come from the same run.
- **Cache-first, credit-conscious FortyGuard usage.** `env_params` is called
  once per day (not per hour) to avoid a known distortion artifact from
  single 24-hour anchoring; `heatmap` is called once per hour (the bulk of
  credit spend, 720 calls for 30 days). `fetch.py` writes progress after
  every completed day so an interrupted run can resume.
- **Physics-based data augmentation.** The 28,800 augmented training rows
  are generated by a heat-balance simulation (`thermal_model.py`), not
  measured telemetry — a deliberate trade-off given the fixed credit
  budget. The model's high R² reflects how well it learned the physics
  formula, not real-building ground truth. Indoor temperature resets to
  23.0 °C each calendar day (no multi-day thermal carryover).
- **Monotonic constraints on the forecaster.** `train_model.py` trains with
  `monotone_constraints` so load is guaranteed non-increasing in setpoint
  and non-decreasing in outdoor/indoor temperature. This exists because an
  earlier model was silently trained on 8 of 10 required features and
  ignored setpoint history entirely — both `train_model.py` and
  `optimizer.py` now hard-fail if a model's feature schema doesn't exactly
  match the expected 10 features.
- **Sequential, thermal-memory load prediction.** `predict_load()`
  reconstructs each day's indoor-temperature trajectory from scratch per
  candidate schedule, so a 10 AM setpoint decision genuinely affects the 2
  PM prediction — the physical basis for pre-cooling strategies.
- **Multi-stage optimization.** Per day: hand-authored templates → Powell
  local polish → differential-evolution global search (multiple restarts)
  → monthly coordinate descent (swaps one day's schedule against the whole
  month's bill, since the demand charge is set by a single peak hour
  across 30 days) → a final peak-refinement pass targeting whichever day
  currently sets the monthly peak.
- **Comfort as a soft, graded cost.** A hard band (occupied 21.5–24.5 °C,
  unoccupied 21.5–25.0 °C) is always enforced; on top of that, an
  ASHRAE-55-*inspired* (not compliant — the standard technically applies to
  naturally-ventilated buildings) adaptive comfort target adds a real-dollar
  quadratic penalty for straying far from the day's target even within the
  hard band, so the optimizer doesn't just park at the comfort ceiling
  every day regardless of weather.

## Known open items

- Stale `data/rag/` (generated before the cache-consistency fix) can show
  numbers that don't match current chart data — delete `data/rag/` to force
  regeneration from `data/cache/`.
- `optimized_comfort_penalty` can end up larger than baseline's — expected,
  since the optimizer is allowed to trade a higher soft comfort cost for
  larger energy/demand savings, not a bug.
- Multi-day thermal carryover is not implemented.

## API

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{status, rag_agent_loaded, days_loaded}` |
| GET | `/api/summary` | Monthly baseline vs. optimized cost/peak/comfort |
| GET | `/api/daily-peak` | Per-day baseline vs. optimized peak kW |
| GET | `/api/dates` | Available dates (`YYYY-MM-DD`) |
| GET | `/api/hourly/{date}` | Hourly setpoint/load/indoor-temp for one day |
| POST | `/api/chat` | `{message}` → `{answer, sources}` via the RAG agent |
| GET | `/` | Dashboard, mounted last so it never shadows `/api/*` |
