"""Shared thermal-state features used by data generation, training and optimization.

v2 — Physics rewrite (see CHANGELOG at bottom of this docstring).

This module builds a small, explicit first-order building thermal model so a
setpoint at hour t can affect HVAC load at t+1 and later hours, and generates
the synthetic training target (`hvac_load_kw`) from an actual heat-balance +
compressor-efficiency model instead of a mostly-flat placeholder.

CHANGELOG (why this file changed):
- The previous version's simulated load was dominated by a large, setpoint-
  independent base term (15 kW out of ~20-30 kW total), so even a perfect
  optimizer could only ever move ~5-8% of the bill. The new heat-balance
  model ties nearly all of the load to the gap between outdoor temperature,
  indoor state, and the setpoint - so setpoint changes now have a real,
  physically-grounded effect on predicted load.
- Added a temperature-dependent compressor efficiency curve (COP degrades
  in extreme heat). This is what makes *timing* matter, not just total
  degree-hours of cooling: running the compressor hard during a 44C
  afternoon costs more electrical kW per unit of heat removed than doing
  the same cooling work in the morning. That is the physical justification
  for pre-cooling and load-shifting, and it stacks with the electricity
  tariff's on-peak/off-peak price difference.
- HVAC_RESPONSE (how fast indoor temperature reacts to a setpoint change)
  was lowered from 0.28 to 0.16/hr. At 0.28 the building re-equilibrated
  in ~2 hours, so a morning pre-cool had mostly faded out before the
  afternoon peak even started. At 0.16 the building has real thermal mass
  (~6 hour time constant), so pre-cooling at 10-11 AM still measurably
  helps at 2-4 PM - which is exactly the strategy the product is meant to
  recommend.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# Bump this whenever the physics/feature-generation logic changes in a way
# that makes previously-generated training data stale. train_model.py checks
# this against a stamped column and forces a rebuild on mismatch, so a stale
# dataset (or a stale model trained on it) can never silently stay deployed
# the way the old 8-feature model did.
THERMAL_MODEL_VERSION = "v2.1-heatbalance"

# ------------------------------------------------------------------
# Building thermal dynamics
# ------------------------------------------------------------------
THERMAL_ALPHA = 0.12        # outdoor -> indoor heat gain per hour (envelope leak)
HVAC_RESPONSE = 0.16        # pull toward the current setpoint per hour
INITIAL_INDOOR_TEMP = 23.0
MIN_INDOOR_TEMP = 18.0
MAX_INDOOR_TEMP = 32.0

# ------------------------------------------------------------------
# Heat-balance coefficients (kW per unit driver)
# ------------------------------------------------------------------
ENVELOPE_UA_KW_PER_C = 1.35   # conduction/infiltration gain per (outdoor-indoor) C
PULLDOWN_KW_PER_C = 1.60      # extra active cooling to pull indoor down to setpoint
SOLAR_KW_PER_WM2 = 0.0040     # solar heat gain per W/m^2 of GHI
OCCUPANCY_GAIN_KW = 1.10      # internal heat gain (people + equipment) at full occupancy
LATENT_KW_PER_PCT_RH = 0.0060 # latent/dehumidification load per % relative humidity
FAN_BASE_KW = 3.0             # ventilation/fan floor, mostly setpoint-independent

# ------------------------------------------------------------------
# Compressor efficiency (COP) degrades as outdoor temperature rises.
# This is what makes WHEN you cool matter, not just how much.
# ------------------------------------------------------------------
COP_MAX = 4.2
COP_MIN = 1.8
COP_REF_TEMP_C = 25.0   # COP stays at COP_MAX at/below this outdoor temp
COP_SLOPE = 0.095       # COP lost per degree above COP_REF_TEMP_C


def compressor_cop(outdoor_temp_c: np.ndarray) -> np.ndarray:
    """Electrical efficiency of the cooling plant as a function of ambient temp.

    Real vapor-compression chillers/AC units lose capacity and efficiency as
    the outdoor (condenser-side) temperature rises. Modeling this - even
    simply - is what gives the optimizer a genuine reason to shift cooling
    load away from the hottest hours, beyond just chasing the electricity
    tariff.
    """
    outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)
    cop = COP_MAX - COP_SLOPE * np.maximum(0.0, outdoor_temp_c - COP_REF_TEMP_C)
    return np.clip(cop, COP_MIN, COP_MAX)


def build_thermal_features(
    day_df: pd.DataFrame,
    setpoints: list[float] | np.ndarray,
    initial_indoor_temp: float = INITIAL_INDOOR_TEMP,
) -> pd.DataFrame:
    """Create sequential thermal features for one chronological day.

    `indoor_temp_c` is the indoor state at the BEGINNING of the hour (before
    that hour's HVAC response is applied). `previous_setpoint_c` is the
    setpoint that was active during the previous hour.

    State transition per hour:
        indoor += THERMAL_ALPHA * (outdoor - indoor)   # passive envelope gain
        indoor += HVAC_RESPONSE * (setpoint - indoor)  # active HVAC pull

    A lower setpoint now genuinely leaves a lower indoor temperature for the
    next several hours (HVAC_RESPONSE=0.16 gives a multi-hour time constant),
    which is the thermal memory a pre-cooling strategy depends on.
    """
    df = day_df.copy().sort_values("timestamp").reset_index(drop=True)
    sp = np.asarray(setpoints, dtype=float)

    if len(df) != len(sp):
        raise ValueError("setpoints length must equal day_df length")

    indoor = float(initial_indoor_temp)
    prev_sp = float(sp[0])  # no previous action is known at the first hour
    indoor_before = []
    prev_setpoints = []

    for i, row in df.iterrows():
        outdoor = float(row["temperature_c"])

        indoor_before.append(indoor)
        prev_setpoints.append(prev_sp)

        # First-order heat transfer + HVAC response.
        indoor += THERMAL_ALPHA * (outdoor - indoor)
        indoor += HVAC_RESPONSE * (sp[i] - indoor)
        indoor = float(np.clip(indoor, MIN_INDOOR_TEMP, MAX_INDOOR_TEMP))

        prev_sp = float(sp[i])

    out = df.copy()
    out["indoor_temp_c"] = np.asarray(indoor_before)
    out["previous_setpoint_c"] = np.asarray(prev_setpoints)
    return out


def simulate_target_loads(
    feature_df: pd.DataFrame,
    setpoints: list[float] | np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate the synthetic HVAC target from an explicit heat-balance model.

    load_kw = FAN_BASE_KW * occupancy_gate
            + (sensible_heat_to_remove + latent_heat_to_remove) / COP(outdoor)

    sensible_heat_to_remove =
          ENVELOPE_UA * max(0, outdoor - indoor)        # passive gain
        + PULLDOWN_COEF * max(0, indoor - setpoint)      # active pulldown
        + SOLAR_COEF * ghi
        + OCCUPANCY_GAIN * occupancy_factor

    Unlike a flat per-hour cost, this ties the electrical load to (a) the
    real-time gap between outdoor/indoor/setpoint, so setpoint changes have
    a first-order effect, and (b) the compressor's COP curve, so *when* the
    cooling happens changes how many kWh it costs even for the same amount
    of heat removed - which is the whole justification for pre-cooling and
    load-shifting.
    """
    sp = np.asarray(setpoints, dtype=float)
    indoor = feature_df["indoor_temp_c"].to_numpy(dtype=float)
    outdoor = feature_df["temperature_c"].to_numpy(dtype=float)
    humidity = feature_df["humidity_pct"].to_numpy(dtype=float)
    ghi = feature_df["ghi_wm2"].to_numpy(dtype=float)
    occ = feature_df["occupancy_factor"].to_numpy(dtype=float)

    envelope_gain = ENVELOPE_UA_KW_PER_C * np.maximum(0.0, outdoor - indoor)
    pulldown = PULLDOWN_KW_PER_C * np.maximum(0.0, indoor - sp)
    solar_gain = SOLAR_KW_PER_WM2 * np.maximum(0.0, ghi)
    occupancy_gain = OCCUPANCY_GAIN_KW * occ

    sensible = envelope_gain + pulldown + solar_gain + occupancy_gain
    latent = LATENT_KW_PER_PCT_RH * humidity * np.maximum(occ, 0.15)

    cop = compressor_cop(outdoor)
    cooling_electrical_kw = (sensible + latent) / cop

    fan_kw = FAN_BASE_KW * np.maximum(occ, 0.20)

    load = fan_kw + cooling_electrical_kw

    if noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        load = load + rng.normal(0.0, noise_std, size=load.shape)

    return np.maximum(load, 0.0)


def make_dynamic_training_rows(
    weather_df: pd.DataFrame,
    schedules_per_day: int = 60,
    seed: int = 42,
    noise_std: float = 0.35,
) -> pd.DataFrame:
    """Create many sequential control trajectories from historical weather.

    Includes:
      - A handful of flat baselines (for calibration).
      - A dense set of DETERMINISTIC pre-cool / peak-shave templates that
        deliberately mirror the shapes optimizer.py's
        generate_candidate_schedules() searches over. If the training
        distribution doesn't cover the exact region the optimizer searches,
        a tree model's predictions there are extrapolation-prone and can
        look artificially flat even when the underlying physics isn't -
        so we make sure that region is densely and explicitly sampled.
      - Randomized piecewise trajectories for general coverage of the
        24-hour setpoint space.

    A small amount of Gaussian noise is added to the simulated target so the
    trained model doesn't fit a perfectly noiseless deterministic function
    (more realistic, and a fairer test of the optimizer against a model that
    behaves like a real forecaster).
    """
    rng = np.random.default_rng(seed)
    rows = []

    setpoint_choices = np.arange(21.5, 25.01, 0.25)

    for _, day in weather_df.groupby(
        pd.to_datetime(weather_df["timestamp"]).dt.date, sort=True
    ):
        day = day.sort_values("timestamp").reset_index(drop=True)
        if len(day) == 0:
            continue

        hours = day["hour"].to_numpy()

        # ---- Flat baselines, for calibration ----
        schedules = [
            np.full(len(day), 23.0),
            np.full(len(day), 25.0),
            np.full(len(day), 22.0),
            np.full(len(day), 21.5),
        ]

        # ---- Deterministic templates mirroring the optimizer's search
        #      space, so the model sees dense, labeled examples in exactly
        #      the region it will be asked to evaluate. ----
        for precool_start in (9, 10, 11):
            for pre_temp in (21.5, 22.0, 22.5):
                for peak_temp in (24.0, 24.5, 25.0):
                    sp = np.empty(len(day), dtype=float)
                    for i, h in enumerate(hours):
                        if precool_start <= h < 12:
                            sp[i] = pre_temp
                        elif 12 <= h < 14:
                            sp[i] = 22.5
                        elif 14 <= h < 19:
                            sp[i] = peak_temp
                        else:
                            sp[i] = 23.0
                    schedules.append(sp)

        for peak_temp in (24.0, 24.25, 24.5, 24.75, 25.0):
            sp = np.array(
                [peak_temp if 14 <= h < 19 else 23.0 for h in hours],
                dtype=float,
            )
            schedules.append(sp)

        # ---- Randomized piecewise trajectories for general coverage ----
        n_random = max(0, schedules_per_day - len(schedules))
        for _ in range(n_random):
            sp = np.empty(len(day), dtype=float)
            current = float(rng.choice(setpoint_choices))
            for i in range(len(day)):
                if i == 0 or rng.random() < 0.22:
                    current = float(rng.choice(setpoint_choices))
                sp[i] = current
            schedules.append(sp)

        for schedule in schedules:
            features = build_thermal_features(day, schedule)
            loads = simulate_target_loads(
                features, schedule, noise_std=noise_std, rng=rng
            )

            for i, row in features.iterrows():
                r = row.to_dict()
                r["comfort_setpoint_c"] = float(schedule[i])
                r["hvac_load_kw"] = float(loads[i])
                r["_thermal_version"] = THERMAL_MODEL_VERSION
                rows.append(r)

    return pd.DataFrame(rows)