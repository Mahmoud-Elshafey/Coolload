from __future__ import annotations

import os
import time
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, differential_evolution
from thermal_model import build_thermal_features


# ============================================================
# Electricity Tariff
# ============================================================

OFF_PEAK_RATE = 0.09        # $/kWh
ON_PEAK_RATE = 0.26         # $/kWh
DEMAND_CHARGE_RATE = 14.50  # $/kW/month

# ============================================================
# Adaptive Comfort Model (ASHRAE 55 - inspired, with TWO caveats)
# ============================================================
#
# CAVEAT 1: ASHRAE 55's adaptive comfort method, strictly read, applies
# to naturally-ventilated spaces where occupants control their own
# openings - not to centrally air-conditioned commercial buildings
# like the one this product targets. We deliberately borrow the
# adaptive comfort MODEL - people acclimate to the recent outdoor
# climate, so the temperature that feels "neutral" shifts with it -
# as a well-established, defensible basis for a day-by-day comfort
# target. This is NOT a claim of formal ASHRAE 55 compliance; say so
# plainly if asked in a pitch or technical review.
#
# CAVEAT 2 (found while building this - important): ASHRAE 55's own
# coefficients (Tcomfort = 0.31*Trm + 17.8) were fit to TEMPERATE
# climates. Solving for where that formula alone already reaches a
# typical 24.5C mechanical-cooling ceiling gives Trm = 21.6C - and a
# hot-climate commercial building's running-mean outdoor temperature
# sits above that on nearly every single day of the year. Used
# literally, the standard's own numbers would say "always run at the
# ceiling," permanently - which defeats the entire point of an
# adaptive target and would have silently reproduced the exact
# "always max setpoint" problem we're trying to fix here.
#
# The fix: keep ASHRAE 55's MECHANISM (an exponentially-weighted
# running mean of recent outdoor temperature, alpha=0.8, exactly as
# the standard defines it - real day-to-day acclimatization behavior)
# but replace its fixed temperate-climate slope/intercept with a
# PERCENTILE-BASED RESCALE onto this building's own achievable
# comfort band: the coolest ~10th-percentile running-mean days in the
# dataset map near the comfort floor, the hottest ~90th-percentile
# days map near the ceiling, and everything in between is
# interpolated. This is climate-relative by construction, so it
# produces genuine day-to-day variation whether the building sits in
# a desert or a mild coastal city, instead of saturating immediately
# in any hot-climate deployment.
ADAPTIVE_COMFORT_ALPHA = 0.8          # ASHRAE 55's own running-mean smoothing constant
ADAPTIVE_COMFORT_PCTL_LOW = 10.0      # Trm percentile mapped to the comfort floor
ADAPTIVE_COMFORT_PCTL_HIGH = 90.0     # Trm percentile mapped to the comfort ceiling

# Soft comfort penalty (real $, added to the objective):
# no cost inside a small deadband around the day's adaptive target;
# quadratic cost beyond it. This is what stops the optimizer from
# parking at the exact top of the hard comfort band every day
# regardless of weather - the previous behavior with comfort_penalty
# hardcoded to a hard hard-wall-only, zero-inside constraint.
COMFORT_DEADBAND_C = 0.5
COMFORT_PENALTY_RATE = 0.10   # $ per (deg beyond deadband)^2 per occupied hour
UNOCCUPIED_COMFORT_WEIGHT = 0.15


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(PROJECT_ROOT / "models" / "hvac_forecaster.joblib")
DATA_PATH = str(PROJECT_ROOT / "data" / "hvac_30day_dataset.csv")


class HVACOptimizer:

    FEATURE_COLS = [
        "temperature_c",
        "humidity_pct",
        "ghi_wm2",
        "hour",
        "day_of_week",
        "is_weekend",
        "occupancy_factor",
        "comfort_setpoint_c",
        "previous_setpoint_c",
        "indoor_temp_c",
    ]

    # Hard comfort-band bounds shared by every optimizer
    # (discrete candidate search, continuous polish, DE, peak refinement).
    SETPOINT_MIN = 21.5
    SETPOINT_MAX = 25.0

    # Hard comfort policy. Occupancy >= 0.50 is considered occupied.
    OCCUPANCY_THRESHOLD = 0.50
    OCCUPIED_SETPOINT_MIN = 21.5
    OCCUPIED_SETPOINT_MAX = 24.5
    UNOCCUPIED_SETPOINT_MIN = 21.5
    UNOCCUPIED_SETPOINT_MAX = 25.0

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        comfort_penalty_rate: float = COMFORT_PENALTY_RATE,
        enable_continuous_search: bool = True,
        polish_top_n: int = 4,
        polish_maxiter: int = 200,
        de_maxiter: int = 60,
        de_popsize: int = 10,
        de_restarts: int = 2,
        peak_refine_rounds: int = 12,
        peak_search_weight: float = 6.0,
        random_seed: int = 42,
        parallel_days: bool = True,
        max_workers: int | None = None
    ):
        """
        Parameters
        ----------
        enable_continuous_search:
            If True, every day's discrete candidate pool is
            augmented with continuous, gradient-free local optimization
            (Powell polish of the strongest discrete candidates) and a
            global differential-evolution search over the full 24-hour
            setpoint vector. This is what lets the optimizer escape the
            small set of hand-authored templates and find schedules the
            templates can't express (asymmetric pre-cool ramps, partial
            setback hours, etc.). Defaults to True: the discrete
            templates alone consistently leave real savings on the
            table, and this is the single biggest lever for closing the
            gap to the true achievable optimum.
        polish_top_n:
            How many of the best discrete candidates to locally refine
            with a bounded Powell search each day.
        polish_maxiter / de_maxiter / de_popsize:
            Effort knobs for the local polish and the differential
            evolution global search. Raise these for a slower but more
            thorough search; lower them for faster (but slightly weaker)
            results.
        de_restarts:
            Number of independent differential-evolution runs per day,
            each from a different random seed. Real forecasting models
            can have a bumpy, non-convex response to setpoint changes;
            a single DE run can land in a mediocre local optimum, so
            multiple restarts (keeping only the best) meaningfully
            raise the odds of finding the true best schedule for the day.
        peak_refine_rounds:
            Number of monthly demand-charge-focused refinement rounds
            run after the coordinate-descent global optimizer converges.
            Each round retargets whichever single day is currently
            setting the monthly peak (and therefore the demand charge)
            and tries to shave it down specifically.
        peak_search_weight:
            How strongly the peak-refinement stage prioritizes cutting
            the targeted day's peak load, on top of the hard peak_cap
            penalty. Higher values push harder toward flattening that
            day even for a small energy-cost trade-off.
        parallel_days:
            If True, each calendar day's candidate pool (discrete grid
            + Powell polish + DE global search) is built concurrently
            across a thread pool instead of sequentially. Days are
            fully independent at this stage, so this is "free" wall-
            clock speedup that lets the heavier de_maxiter/de_popsize/
            de_restarts settings above run in roughly the same total
            time as the old, weaker defaults used to take serially.
        max_workers:
            Thread pool size for parallel_days. Defaults to
            min(32, os.cpu_count() + 4), matching
            ThreadPoolExecutor's own default.
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Train model first."
            )

        self.model = joblib.load(model_path)

        # ------------------------------------------------------------
        # HARD SCHEMA GUARD.
        #
        # A previously-deployed model was silently trained on only 8 of
        # the 10 FEATURE_COLS (missing previous_setpoint_c and
        # indoor_temp_c). predict_load() still built those columns
        # correctly, but the model ignored them - so every hour was
        # scored independently with zero thermal memory, and pre-cooling
        # / load-shifting strategies could never show a benefit. That
        # was silent: no exception, no warning, just quietly worse
        # savings. This check turns that failure mode into an immediate,
        # loud error instead of a silent capability loss.
        # ------------------------------------------------------------
        get_booster = getattr(self.model, "get_booster", None)
        if get_booster is not None:
            booster_features = get_booster().feature_names
            if (
                booster_features is not None
                and list(booster_features) != list(self.FEATURE_COLS)
            ):
                missing = set(self.FEATURE_COLS) - set(booster_features)
                extra = set(booster_features) - set(self.FEATURE_COLS)
                raise ValueError(
                    "Loaded model's feature schema does not match "
                    "HVACOptimizer.FEATURE_COLS - refusing to run with a "
                    "mismatched model.\n"
                    f"  Model was trained on : {booster_features}\n"
                    f"  Optimizer expects    : {self.FEATURE_COLS}\n"
                    f"  Missing from model   : {sorted(missing) or 'none'}\n"
                    f"  Extra in model        : {sorted(extra) or 'none'}\n"
                    "Retrain with train_model.py using the current "
                    "thermal_model.py, then reload."
                )

        # Comfort is now a graded, real-dollar soft cost (see the
        # Adaptive Comfort Model constants above) layered on top of
        # the hard comfort-band constraint, not a hardcoded no-op.
        self.comfort_penalty_rate = comfort_penalty_rate

        self.enable_continuous_search = enable_continuous_search
        self.polish_top_n = polish_top_n
        self.polish_maxiter = polish_maxiter
        self.de_maxiter = de_maxiter
        self.de_popsize = de_popsize
        self.de_restarts = de_restarts
        self.peak_refine_rounds = peak_refine_rounds
        self.peak_search_weight = peak_search_weight
        self.random_seed = random_seed
        self.parallel_days = parallel_days
        self.max_workers = max_workers

    # ========================================================
    # 1. Energy Cost
    # ========================================================

    def calculate_energy_cost(
        self,
        loads_kw: np.ndarray,
        hours: np.ndarray
    ) -> float:

        hourly_costs = []

        for kw, hour in zip(loads_kw, hours):

            rate = (
                ON_PEAK_RATE
                if 14 <= hour < 19
                else OFF_PEAK_RATE
            )

            # Each row = 1 hour
            hourly_costs.append(max(0.0, kw) * rate)

        return float(np.sum(hourly_costs))

    # ========================================================
    # 2. Monthly Demand Charge
    # ========================================================

    def calculate_demand_charge(
        self,
        loads_kw: np.ndarray
    ) -> float:

        monthly_peak_kw = float(np.max(loads_kw))

        return monthly_peak_kw * DEMAND_CHARGE_RATE

    # ========================================================
    # 3. Comfort Penalty
    # ========================================================

    def get_setpoint_bounds(
        self,
        occupancy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the hard setpoint bounds for each hour."""
        occupancy = np.asarray(occupancy, dtype=float)
        occupied = occupancy >= self.OCCUPANCY_THRESHOLD

        lower = np.where(
            occupied,
            self.OCCUPIED_SETPOINT_MIN,
            self.UNOCCUPIED_SETPOINT_MIN
        )
        upper = np.where(
            occupied,
            self.OCCUPIED_SETPOINT_MAX,
            self.UNOCCUPIED_SETPOINT_MAX
        )
        return lower, upper

    def validate_setpoints(
        self,
        setpoints: list[float],
        occupancy: np.ndarray
    ) -> bool:
        """Hard comfort feasibility check; invalid schedules are rejected."""
        setpoints = np.asarray(setpoints, dtype=float)
        lower, upper = self.get_setpoint_bounds(occupancy)

        if len(setpoints) != len(lower):
            return False

        return bool(
            np.all(setpoints >= lower - 1e-9)
            and np.all(setpoints <= upper + 1e-9)
        )

    def compute_adaptive_comfort_targets(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Compute a per-row adaptive comfort target (deg C), one value
        per calendar day: an ASHRAE-55-style running mean of outdoor
        temperature, rescaled onto this building's own comfort band
        by percentile (see the module-level comment above for why a
        straight application of ASHRAE 55's own coefficients
        saturates immediately in a hot climate, and why percentile
        rescaling is used instead).

        Trm_n = (1 - alpha) * mean_outdoor_temp(day n-1) + alpha * Trm_(n-1)

        The first day in the dataset has no prior history and is
        seeded with its own daily mean outdoor temperature (a
        reasonable fallback when no earlier weather history is
        available).

        The coolest ADAPTIVE_COMFORT_PCTL_LOW-percentile running-mean
        day in the dataset maps to OCCUPIED_SETPOINT_MIN; the hottest
        ADAPTIVE_COMFORT_PCTL_HIGH-percentile day maps to
        OCCUPIED_SETPOINT_MAX; everything else is linearly
        interpolated between them.
        """
        daily_mean = (
            df.groupby(df["timestamp"].dt.date)["temperature_c"]
            .mean()
            .sort_index()
        )

        dates = list(daily_mean.index)
        trm_by_date: dict = {}

        if dates:
            trm = float(daily_mean.iloc[0])
            trm_by_date[dates[0]] = trm

            for i in range(1, len(dates)):
                prev_day_mean = float(daily_mean.iloc[i - 1])
                trm = (
                    (1 - ADAPTIVE_COMFORT_ALPHA) * prev_day_mean
                    + ADAPTIVE_COMFORT_ALPHA * trm
                )
                trm_by_date[dates[i]] = trm

        trm_values = np.array(list(trm_by_date.values()), dtype=float)

        if len(trm_values) == 0:
            trm_low, trm_high = 20.0, 30.0
        else:
            trm_low = float(
                np.percentile(trm_values, ADAPTIVE_COMFORT_PCTL_LOW)
            )
            trm_high = float(
                np.percentile(trm_values, ADAPTIVE_COMFORT_PCTL_HIGH)
            )
            if trm_high - trm_low < 1e-6:
                # A dataset with almost no day-to-day temperature
                # variation (e.g. a very short or unusually uniform
                # window) - fall back to a fixed +/-3C spread around
                # the single observed value so the mapping stays
                # well-defined instead of dividing by ~0.
                trm_high = trm_low + 3.0

        band_width = (
            self.OCCUPIED_SETPOINT_MAX - self.OCCUPIED_SETPOINT_MIN
        )

        def _target_for_date(d):
            trm = trm_by_date.get(d, trm_low)
            frac = (trm - trm_low) / (trm_high - trm_low)
            frac = float(np.clip(frac, 0.0, 1.0))
            return self.OCCUPIED_SETPOINT_MIN + frac * band_width

        return df["timestamp"].dt.date.map(_target_for_date)

    def calculate_comfort_penalty(
        self,
        setpoints: list[float],
        occupancy: np.ndarray,
        comfort_target_c: np.ndarray | float | None = None
    ) -> float:
        """
        Hard-band violation (always checked) PLUS a graded soft
        penalty against that day's adaptive comfort target (only
        when comfort_target_c is supplied).

        Hard violation: schedules outside the occupied/unoccupied
        band are rejected upstream by validate_setpoints(), so this
        term is a safety net and is ~0 for every candidate that
        actually reaches this function in practice.

        Soft penalty (real $, this is what changed): even INSIDE the
        hard band, sitting more than COMFORT_DEADBAND_C away from the
        day's adaptive comfort target now costs something, growing
        quadratically with distance. Unoccupied hours are weighted
        down (occupant comfort barely matters when no one is there)
        rather than dropped entirely, since a building shouldn't
        swing to a wild extreme even when empty.
        """
        setpoints = np.asarray(setpoints, dtype=float)
        occupancy = np.asarray(occupancy, dtype=float)
        lower, upper = self.get_setpoint_bounds(occupancy)

        low_violation = np.maximum(lower - setpoints, 0.0)
        high_violation = np.maximum(setpoints - upper, 0.0)
        hard_violation = float(np.sum(low_violation + high_violation))

        if comfort_target_c is None:
            return hard_violation

        target = np.broadcast_to(
            np.asarray(comfort_target_c, dtype=float),
            setpoints.shape
        )

        deviation = np.abs(setpoints - target)
        excess = np.maximum(deviation - COMFORT_DEADBAND_C, 0.0)

        occupied = occupancy >= self.OCCUPANCY_THRESHOLD
        weight = np.where(occupied, 1.0, UNOCCUPIED_COMFORT_WEIGHT)

        soft_penalty = float(
            np.sum(weight * self.comfort_penalty_rate * (excess ** 2))
        )

        # Hard violations should never occur here in practice (they're
        # rejected upstream), but if they somehow do, they must swamp
        # the soft term rather than being masked by it.
        return hard_violation * 1000.0 + soft_penalty

    # ========================================================
    # 4. Monthly Objective
    # ========================================================

    def calculate_objective(
        self,
        loads_kw: np.ndarray,
        hours: np.ndarray,
        setpoints: list[float],
        occupancy: np.ndarray,
        comfort_target_c: np.ndarray | float | None = None
    ) -> dict:

        energy_cost = self.calculate_energy_cost(
            loads_kw,
            hours
        )

        peak_kw = float(np.max(loads_kw))

        demand_charge = (
            peak_kw * DEMAND_CHARGE_RATE
        )

        comfort_penalty = self.calculate_comfort_penalty(
            setpoints,
            occupancy,
            comfort_target_c
        )

        # Comfort is now a graded real-dollar cost layered on top of
        # the (still hard) comfort-band constraint, so the search
        # genuinely trades it off against energy/demand savings
        # instead of always maxing out the setpoint for free.
        total_cost = energy_cost + demand_charge + comfort_penalty

        return {
            "energy_cost": energy_cost,
            "peak_kw": peak_kw,
            "demand_charge": demand_charge,
            "comfort_penalty": comfort_penalty,
            "objective": total_cost
        }

    # ========================================================
    # 5. Total Monthly Cost
    # ========================================================

    def calculate_monthly_cost(
        self,
        loads_kw: np.ndarray,
        hours: np.ndarray
    ) -> dict:

        energy_cost = self.calculate_energy_cost(
            loads_kw,
            hours
        )

        monthly_peak_kw = float(np.max(loads_kw))

        demand_charge = self.calculate_demand_charge(
            loads_kw
        )

        total_cost = energy_cost + demand_charge

        return {
            "energy_cost": round(energy_cost, 2),
            "demand_charge": round(demand_charge, 2),
            "total_cost": round(total_cost, 2),
            "peak_kw": round(monthly_peak_kw, 2)
        }

    # ========================================================
    # 6. Predict HVAC Load
    # ========================================================

    def predict_load(
        self,
        df: pd.DataFrame,
        setpoints: list[float]
    ) -> np.ndarray:
        """Predict HVAC load sequentially so setpoints have thermal memory.

        The old implementation sent each hour independently to the model.
        This version first reconstructs the building thermal state for each
        calendar day. Therefore a lower setpoint at 10:00 can cool the
        indoor state used at 11:00, 12:00, ... and can genuinely support
        pre-cooling/load-shifting strategies.
        """
        features = df.copy()
        features["timestamp"] = pd.to_datetime(features["timestamp"])
        features = features.sort_values("timestamp").reset_index(drop=True)

        setpoints_arr = np.asarray(setpoints, dtype=float)
        if len(features) != len(setpoints_arr):
            raise ValueError("setpoints length must equal df length")

        predictions = np.empty(len(features), dtype=float)

        # Thermal state resets at the beginning of each building day.
        for _, idx in features.groupby(
            features["timestamp"].dt.date, sort=True
        ).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            day = features.iloc[idx].copy()
            day_sp = setpoints_arr[idx]

            day_features = build_thermal_features(day, day_sp)
            day_features["comfort_setpoint_c"] = day_sp

            predictions[idx] = np.asarray(
                self.model.predict(day_features[self.FEATURE_COLS]),
                dtype=float
            )

        return predictions

    # ========================================================
    # 7. Generate Candidate Schedule for ONE DAY
    # ========================================================

    def generate_candidate_schedules(
        self,
        day_df: pd.DataFrame
    ):

        hours = day_df["hour"].values
        occupancy = day_df["occupancy_factor"].values

        schedules = []

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        schedules.append({
            "name": "Baseline",
            "setpoints": [23.0] * len(hours)
        })

        # Maximum setpoint remains a useful benchmark, but it is no longer
        # automatically dominant: predict_load() carries indoor thermal
        # state forward, so earlier pre-cooling can change later loads.
        max_setpoints = [
            self.OCCUPIED_SETPOINT_MAX
            if occ >= self.OCCUPANCY_THRESHOLD
            else self.UNOCCUPIED_SETPOINT_MAX
            for occ in occupancy
        ]

        schedules.append({
            "name": "Maximum Comfortable Setpoint",
            "setpoints": max_setpoints
        })

        # ----------------------------------------------------
        # Peak Shaving (finer temp grid + variable peak window)
        # ----------------------------------------------------

        peak_temp_grid = [24.0, 24.25, 24.5, 24.75, 25.0]
        peak_window_grid = [
            (13, 18), (13, 19), (13, 20),
            (14, 18), (14, 19), (14, 20),
            (15, 19), (15, 20)
        ]

        for peak_temp in peak_temp_grid:

            schedule = []

            for h in hours:

                if 14 <= h < 19:
                    schedule.append(peak_temp)
                else:
                    schedule.append(23.0)

            schedules.append({
                "name": f"Peak Shaving {peak_temp}C",
                "setpoints": schedule
            })

        for (start, end), peak_temp in itertools.product(
            peak_window_grid,
            [24.5, 25.0]
        ):

            schedule = []

            for h in hours:

                if start <= h < end:
                    schedule.append(peak_temp)
                else:
                    schedule.append(23.0)

            schedules.append({
                "name": (
                    f"Peak Shaving {peak_temp}C "
                    f"[{start}-{end}h]"
                ),
                "setpoints": schedule
            })

        # ----------------------------------------------------
        # Pre-cooling + Peak Shaving (finer temp grid +
        # variable pre-cool start hour)
        # ----------------------------------------------------

        precool_start_grid = [9, 10, 11]

        for precool_start, pre_temp, peak_temp in itertools.product(
            precool_start_grid,
            [21.5, 21.75, 22.0, 22.25, 22.5],
            [24.0, 24.25, 24.5, 24.75, 25.0]
        ):

            schedule = []

            for h in hours:

                if precool_start <= h < 12:
                    schedule.append(pre_temp)

                elif 12 <= h < 14:
                    schedule.append(22.5)

                elif 14 <= h < 19:
                    schedule.append(peak_temp)

                else:
                    schedule.append(23.0)

            schedules.append({
                "name": (
                    f"PreCool[{precool_start}h] {pre_temp}C + "
                    f"Peak {peak_temp}C"
                ),
                "setpoints": schedule
            })

        # ----------------------------------------------------
        # Aggressive Pre-cooling
        # ----------------------------------------------------

        for pre_temp, peak_temp in itertools.product(
            [21.5, 22.0],
            [24.5, 25.0]
        ):

            schedule = []

            for h in hours:

                if 9 <= h < 12:
                    schedule.append(pre_temp)

                elif 12 <= h < 14:
                    schedule.append(22.0)

                elif 14 <= h < 19:
                    schedule.append(peak_temp)

                else:
                    schedule.append(23.0)

            schedules.append({
                "name": (
                    f"Aggressive PreCool {pre_temp}C + "
                    f"Peak {peak_temp}C"
                ),
                "setpoints": schedule
            })

        # ----------------------------------------------------
        # Occupancy Aware
        # ----------------------------------------------------

        occupancy_schedule = []

        for h, occ in zip(hours, occupancy):

            if 14 <= h < 19:

                if occ >= 0.70:
                    occupancy_schedule.append(24.0)

                elif occ >= 0.30:
                    occupancy_schedule.append(24.5)

                else:
                    occupancy_schedule.append(25.0)

            else:

                if occ >= 0.70:
                    occupancy_schedule.append(23.0)

                elif occ >= 0.30:
                    occupancy_schedule.append(23.5)

                else:
                    occupancy_schedule.append(24.0)

        schedules.append({
            "name": "Occupancy Aware",
            "setpoints": occupancy_schedule
        })

        # ----------------------------------------------------
        # Remove duplicate schedules
        # ----------------------------------------------------

        unique = []
        seen = set()

        for schedule in schedules:

            key = tuple(schedule["setpoints"])

            if key not in seen:
                seen.add(key)
                unique.append(schedule)

        return unique

    # ========================================================
    # 8. Evaluate ONE DAY
    # ========================================================

    def evaluate_day_schedule(
        self,
        day_df: pd.DataFrame,
        schedule: dict
    ) -> dict:

        setpoints = schedule["setpoints"]
        occupancy = day_df["occupancy_factor"].values

        # HARD comfort constraint: reject any schedule outside the
        # occupied/unoccupied comfort band.
        if not self.validate_setpoints(setpoints, occupancy):
            return None

        loads = self.predict_load(
            day_df,
            setpoints
        )

        hours = day_df["hour"].values

        comfort_target_c = (
            day_df["adaptive_comfort_target_c"].values
            if "adaptive_comfort_target_c" in day_df.columns
            else None
        )

        metrics = self.calculate_objective(
            loads,
            hours,
            setpoints,
            occupancy,
            comfort_target_c
        )

        return {
            "name": schedule["name"],
            "setpoints": setpoints,
            "loads": loads,
            "_comfort_target": comfort_target_c,
            **metrics
        }

    # ========================================================
    # 9. Energy Optimizer
    # ========================================================

    def optimize_energy_strategy(
        self,
        evaluated: list[dict]
    ) -> dict:

        return min(
            evaluated,
            key=lambda x: x["energy_cost"]
        )

    # ========================================================
    # 10. Peak Optimizer
    # ========================================================

    def optimize_peak_strategy(
        self,
        evaluated: list[dict]
    ) -> dict:

        return min(
            evaluated,
            key=lambda x: (
                x["peak_kw"],
                x["energy_cost"]
            )
        )

    # ========================================================
    # 11. Comfort Optimizer
    # ========================================================

    def optimize_comfort_strategy(
        self,
        evaluated: list[dict]
    ) -> dict:

        return min(
            evaluated,
            key=lambda x: (
                x["comfort_penalty"],
                x["energy_cost"]
            )
        )

    # ========================================================
    # 11b. Continuous Optimization Helpers
    #
    # The template candidates in generate_candidate_schedules()
    # are a small hand-picked grid (fixed windows, 0.5C steps).
    # They're a good, explainable starting point, but they cap
    # how much savings the optimizer can ever find, because the
    # true optimum rarely lands exactly on a hand-picked template.
    #
    # These helpers add genuine numerical optimization on top:
    #   - polish_day_schedule(): bounded local search (Powell)
    #     that fine-tunes an existing schedule hour-by-hour in
    #     continuous space, within the comfort band.
    #   - differential_evolution_day_schedule(): a global,
    #     derivative-free search over the full 24-hour setpoint
    #     vector, so the optimizer isn't limited to the shapes
    #     the templates assume (single pre-cool block, single
    #     peak block, etc).
    # ========================================================

    def _day_objective_from_array(
        self,
        day_df: pd.DataFrame,
        hours: np.ndarray,
        occupancy: np.ndarray,
        setpoints_arr: np.ndarray,
        peak_weight: float = 0.0,
        peak_cap: float | None = None,
        peak_cap_penalty: float = 500.0
    ) -> float:
        """
        Day-level objective used by the continuous optimizers.

        By default this mirrors calculate_objective() (energy cost +
        comfort penalty). Two optional terms let the SAME machinery be
        reused for demand-charge-focused peak shaving:

        - peak_weight: adds a soft cost per kW of that day's peak load,
          nudging the search toward flatter days even when it isn't
          the global monthly peak yet.
        - peak_cap / peak_cap_penalty: adds a steep penalty if this
          day's peak load exceeds `peak_cap`, used to actively push a
          day's peak below the level set by the rest of the month.
        """

        lower, upper = self.get_setpoint_bounds(occupancy)
        setpoints = np.clip(setpoints_arr, lower, upper)

        loads = self.predict_load(day_df, setpoints)

        comfort_target_c = (
            day_df["adaptive_comfort_target_c"].values
            if "adaptive_comfort_target_c" in day_df.columns
            else None
        )

        metrics = self.calculate_objective(
            loads,
            hours,
            setpoints,
            occupancy,
            comfort_target_c
        )

        objective = metrics["objective"]

        if peak_weight > 0.0:
            objective += peak_weight * metrics["peak_kw"]

        if peak_cap is not None:
            excess = max(0.0, metrics["peak_kw"] - peak_cap)
            objective += peak_cap_penalty * excess

        return float(objective)

    def polish_day_schedule(
        self,
        day_df: pd.DataFrame,
        hours: np.ndarray,
        occupancy: np.ndarray,
        initial_setpoints: list[float],
        peak_weight: float = 0.0,
        peak_cap: float | None = None
    ) -> list[float]:
        """
        Locally refines an existing (discrete-template) schedule with
        a bounded, derivative-free local search (Powell), respecting
        the comfort band at every hour. This is cheap relative to a
        full global search and reliably improves on hand-picked
        templates because it can move each hour independently instead
        of following a fixed block shape.
        """

        n_hours = len(initial_setpoints)

        lower, upper = self.get_setpoint_bounds(occupancy)
        bounds = Bounds(lower, upper)

        def objective(x):
            return self._day_objective_from_array(
                day_df, hours, occupancy, x,
                peak_weight=peak_weight,
                peak_cap=peak_cap
            )

        x0 = np.clip(
            np.asarray(initial_setpoints, dtype=float),
            lower,
            upper
        )

        result = minimize(
            objective,
            x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": self.polish_maxiter, "xtol": 1e-3}
        )

        return np.clip(
            result.x,
            self.SETPOINT_MIN,
            self.SETPOINT_MAX
        ).tolist()

    def differential_evolution_day_schedule(
        self,
        day_df: pd.DataFrame,
        hours: np.ndarray,
        occupancy: np.ndarray,
        peak_weight: float = 0.0,
        peak_cap: float | None = None,
        seed_setpoints: list[float] | None = None
    ) -> list[float]:
        """
        Global search over the full 24-hour setpoint vector using
        differential evolution. Unlike the templates (which always
        follow a single pre-cool-block + peak-block shape) and unlike
        polish_day_schedule() (which only locally refines a starting
        point), this can discover fundamentally different schedule
        shapes - e.g. two separate setbacks, an asymmetric ramp, or a
        shorter/longer peak window than any template assumes.

        Runs `self.de_restarts` independent searches (different random
        seeds - one seeded/jittered around the known-good starting
        point, the rest from broader random exploration) and keeps
        whichever run found the lowest objective. A real forecasting
        model's response surface can be bumpy; a single DE run can
        stall in a mediocre basin, and restarts are cheap insurance
        against leaving genuine savings on the table.
        """

        n_hours = len(hours)

        lower, upper = self.get_setpoint_bounds(occupancy)
        bounds = list(zip(lower.tolist(), upper.tolist()))

        def objective(x):
            return self._day_objective_from_array(
                day_df, hours, occupancy, x,
                peak_weight=peak_weight,
                peak_cap=peak_cap
            )

        has_seed = (
            seed_setpoints is not None
            and len(seed_setpoints) == n_hours
        )

        best_x = None
        best_obj = np.inf

        n_restarts = max(1, self.de_restarts)

        for restart_i in range(n_restarts):

            restart_seed = self.random_seed + restart_i

            # First restart (if a seed schedule is available) is
            # seeded/jittered around it so DE polishes a known-good
            # point; remaining restarts explore more broadly via
            # Sobol sampling with a different seed each time, to
            # cover parts of the search space the seeded run won't.
            if restart_i == 0 and has_seed:

                base = np.clip(
                    np.asarray(seed_setpoints, dtype=float),
                    lower,
                    upper
                )
                n_pop = max(self.de_popsize * n_hours, n_hours + 4)
                rng = np.random.default_rng(restart_seed)
                jitter = rng.normal(0.0, 0.4, size=(n_pop, n_hours))
                init = np.clip(
                    base[None, :] + jitter,
                    lower,
                    upper
                )
            else:
                init = "sobol"

            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=self.de_maxiter,
                popsize=self.de_popsize,
                init=init,
                seed=restart_seed,
                tol=1e-3,
                mutation=(0.4, 1.2),
                recombination=0.8,
                polish=True,
                updating="deferred",
                workers=1
            )

            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x

        return np.clip(best_x, lower, upper).tolist()

    # ========================================================
    # 12. Build Day Candidate Pool
    # ========================================================

    def build_day_candidate_pool(
        self,
        day_df: pd.DataFrame
    ) -> list[dict]:

        schedules = self.generate_candidate_schedules(
            day_df
        )

        evaluated = []

        for schedule in schedules:

            result = self.evaluate_day_schedule(
                day_df,
                schedule
            )

            if result is not None:
                evaluated.append(result)

        if not evaluated:
            raise ValueError(
                "No valid HVAC schedules were generated "
                "for one of the days."
            )

        # ----------------------------------------------------
        # Multiple specialized optimizers
        # ----------------------------------------------------

        selected = {
            "energy": self.optimize_energy_strategy(
                evaluated
            ),
            "peak": self.optimize_peak_strategy(
                evaluated
            ),
            "comfort": self.optimize_comfort_strategy(
                evaluated
            )
        }

        # ----------------------------------------------------
        # Keep all generated candidates PLUS the specialist
        # winners. This allows the global optimizer to make
        # the final decision instead of forcing one optimizer.
        # ----------------------------------------------------

        pool = list(evaluated)

        existing = {
            tuple(item["setpoints"])
            for item in pool
        }

        for item in selected.values():

            key = tuple(item["setpoints"])

            if key not in existing:
                pool.append(item)
                existing.add(key)

        # ----------------------------------------------------
        # Continuous optimization on top of the discrete grid.
        #
        # The templates above are a fixed set of shapes. To get
        # anywhere near the true best achievable savings, the
        # optimizer needs to search continuous setpoint space
        # too - both by polishing the strongest templates and
        # by running a from-scratch global search (DE) that can
        # find shapes no template expresses.
        # ----------------------------------------------------

        if self.enable_continuous_search:

            hours = day_df["hour"].values
            occupancy = day_df["occupancy_factor"].values

            top_candidates = sorted(
                evaluated,
                key=lambda x: x["objective"]
            )[: self.polish_top_n]

            for rank, candidate in enumerate(top_candidates):

                polished_setpoints = self.polish_day_schedule(
                    day_df,
                    hours,
                    occupancy,
                    candidate["setpoints"]
                )

                polished_result = self.evaluate_day_schedule(
                    day_df,
                    {
                        "name": f"Polished[{rank}]({candidate['name']})",
                        "setpoints": polished_setpoints
                    }
                )

                if polished_result is not None:

                    key = tuple(polished_result["setpoints"])

                    if key not in existing:
                        pool.append(polished_result)
                        existing.add(key)

            # Global differential-evolution search, seeded around
            # the current best candidate so it spends its budget
            # refining a strong starting point rather than
            # exploring purely at random.
            best_seed = min(
                evaluated,
                key=lambda x: x["objective"]
            )["setpoints"]

            try:

                de_setpoints = (
                    self.differential_evolution_day_schedule(
                        day_df,
                        hours,
                        occupancy,
                        seed_setpoints=best_seed
                    )
                )

                de_result = self.evaluate_day_schedule(
                    day_df,
                    {
                        "name": "Global Search (DE)",
                        "setpoints": de_setpoints
                    }
                )

                if de_result is not None:

                    key = tuple(de_result["setpoints"])

                    if key not in existing:
                        pool.append(de_result)
                        existing.add(key)

            except Exception:
                # Continuous search is a bonus on top of the
                # discrete grid - if it fails for any reason
                # (e.g. solver numerical issues), the discrete
                # + polished candidates still give a valid,
                # already-improved pool.
                pass

        return pool

    # ========================================================
    # 13. Evaluate FULL MONTH for a strategy combination
    # ========================================================

    def evaluate_month_plan(
        self,
        day_candidates: list[list[dict]],
        selected_indices: list[int]
    ) -> dict:

        all_loads = []
        all_hours = []
        all_setpoints = []
        all_occupancy = []
        all_comfort_targets = []

        for day_index, candidate_index in enumerate(
            selected_indices
        ):

            candidate = day_candidates[
                day_index
            ][candidate_index]

            all_loads.extend(candidate["loads"])
            all_setpoints.extend(candidate["setpoints"])

        # The candidate dictionaries already contain loads,
        # but hours/occupancy must be reconstructed by the
        # stored metadata below.
        #
        # They are attached to the candidate by optimize_month.
        for day_index, candidate_index in enumerate(
            selected_indices
        ):

            candidate = day_candidates[
                day_index
            ][candidate_index]

            all_hours.extend(candidate["_hours"])
            all_occupancy.extend(candidate["_occupancy"])
            all_comfort_targets.extend(candidate["_comfort_target"])

        loads = np.asarray(all_loads, dtype=float)
        hours = np.asarray(all_hours)
        setpoints = list(all_setpoints)
        occupancy = np.asarray(all_occupancy, dtype=float)
        comfort_targets = np.asarray(all_comfort_targets, dtype=float)

        metrics = self.calculate_objective(
            loads,
            hours,
            setpoints,
            occupancy,
            comfort_targets
        )

        return {
            "loads": loads,
            "setpoints": setpoints,
            "hours": hours,
            "occupancy": occupancy,
            **metrics
        }

    # ========================================================
    # 14. Global Monthly Optimizer
    # ========================================================

    def global_monthly_optimizer(
        self,
        day_candidates: list[list[dict]]
    ) -> tuple[list[int], dict]:
        """
        Coordinate-descent global optimizer.

        Each specialized optimizer proposes strong daily
        candidates, then this optimizer evaluates the COMPLETE
        monthly bill after changing one day.

        Therefore the monthly demand charge is part of the
        optimization decision, not an afterthought.
        """

        # ----------------------------------------------------
        # Start from the true baseline strategy for every day
        # ----------------------------------------------------

        selected_indices = []

        for candidates in day_candidates:

            baseline_index = min(
                range(len(candidates)),
                key=lambda i: (
                    0 if candidates[i]["name"] == "Baseline"
                    else 1,
                    candidates[i]["objective"]
                )
            )

            selected_indices.append(baseline_index)

        current_plan = self.evaluate_month_plan(
            day_candidates,
            selected_indices
        )

        max_iterations = 50

        for iteration in range(max_iterations):

            improved = False

            # Search the best change across all days.
            best_move = None
            best_objective = current_plan["objective"]

            for day_index, candidates in enumerate(
                day_candidates
            ):

                current_index = selected_indices[
                    day_index
                ]

                for candidate_index in range(
                    len(candidates)
                ):

                    if candidate_index == current_index:
                        continue

                    trial_indices = selected_indices.copy()
                    trial_indices[day_index] = candidate_index

                    trial_plan = self.evaluate_month_plan(
                        day_candidates,
                        trial_indices
                    )

                    if (
                        trial_plan["objective"]
                        < best_objective - 1e-9
                    ):

                        best_objective = (
                            trial_plan["objective"]
                        )

                        best_move = (
                            day_index,
                            candidate_index,
                            trial_plan
                        )

            if best_move is None:
                break

            day_index, candidate_index, trial_plan = (
                best_move
            )

            selected_indices[
                day_index
            ] = candidate_index

            current_plan = trial_plan
            improved = True

            if not improved:
                break

        return selected_indices, current_plan

    # ========================================================
    # 14b. Demand-Charge-Focused Peak Refinement
    # ========================================================

    def refine_monthly_peak(
        self,
        day_candidates: list[list[dict]],
        day_dfs: list[pd.DataFrame],
        selected_indices: list[int],
        current_plan: dict
    ) -> tuple[list[int], dict]:
        """
        Iteratively targets whichever single day currently sets the
        monthly peak load (and therefore the full monthly demand
        charge) and runs a continuous, peak-weighted local search
        JUST for that day, capped against the second-highest day's
        peak. If the refined day genuinely lowers the monthly bill,
        it's adopted and the process repeats - since flattening the
        worst day can promote a different day to "new worst".

        This targets the demand charge directly instead of relying on
        the coordinate descent to stumble onto it indirectly, which
        matters because DEMAND_CHARGE_RATE is applied once to a
        single peak hour, so it's cheap to search precisely but easy
        for a generic day-by-day search to under-optimize.
        """

        selected_indices = list(selected_indices)

        if len(day_candidates) < 2:
            # Nothing to compare the peak day against - the demand
            # charge is just that single day's peak either way.
            return selected_indices, current_plan

        for _round in range(self.peak_refine_rounds):

            day_peaks = [
                float(np.max(
                    day_candidates[d][selected_indices[d]]["loads"]
                ))
                for d in range(len(day_candidates))
            ]

            worst_day = int(np.argmax(day_peaks))
            other_peaks = (
                day_peaks[:worst_day] + day_peaks[worst_day + 1:]
            )

            # Cap the worst day's peak just under the current
            # runner-up, so the search has a concrete, achievable
            # target instead of only a vague "go lower" signal.
            target_cap = (
                max(other_peaks) if other_peaks else 0.0
            )

            current_candidate = day_candidates[worst_day][
                selected_indices[worst_day]
            ]

            hours_day = current_candidate["_hours"]
            occupancy_day = current_candidate["_occupancy"]
            day_df = day_dfs[worst_day]

            try:
                refined_setpoints = self.polish_day_schedule(
                    day_df,
                    hours_day,
                    occupancy_day,
                    current_candidate["setpoints"],
                    peak_weight=self.peak_search_weight,
                    peak_cap=target_cap
                )
            except Exception:
                break

            refined_result = self.evaluate_day_schedule(
                day_df,
                {
                    "name": "PeakRefined",
                    "setpoints": refined_setpoints
                }
            )

            if refined_result is None:
                break

            refined_result["_hours"] = hours_day
            refined_result["_occupancy"] = occupancy_day

            day_candidates[worst_day].append(refined_result)
            trial_index = len(day_candidates[worst_day]) - 1

            trial_indices = selected_indices.copy()
            trial_indices[worst_day] = trial_index

            trial_plan = self.evaluate_month_plan(
                day_candidates,
                trial_indices
            )

            if trial_plan["objective"] < current_plan["objective"] - 1e-9:
                selected_indices = trial_indices
                current_plan = trial_plan
            else:
                # This round didn't help the monthly bill overall
                # (e.g. the peak dropped but energy cost rose more
                # than the demand charge saved) - stop rather than
                # keep chasing a shrinking or non-existent gain.
                break

        return selected_indices, current_plan

    # ========================================================
    # 15. Optimize FULL MONTH
    # ========================================================

    def optimize_month(
        self,
        df: pd.DataFrame
    ) -> dict:

        timing = {}
        t_total_start = time.perf_counter()

        if df.empty:
            raise ValueError("Dataset is empty.")

        df = df.copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # Attach the per-row ASHRAE-55-inspired adaptive comfort
        # target BEFORE any day-slicing, so every downstream day_df
        # (discrete candidates, Powell polish, DE, peak refinement)
        # automatically carries it as a column.
        df["adaptive_comfort_target_c"] = (
            self.compute_adaptive_comfort_targets(df)
        )

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        t0 = time.perf_counter()

        baseline_setpoints = [
            23.0
        ] * len(df)

        baseline_loads = self.predict_load(
            df,
            baseline_setpoints
        )

        hours = df["hour"].values
        occupancy = df["occupancy_factor"].values

        baseline_cost = self.calculate_monthly_cost(
            baseline_loads,
            hours
        )

        baseline_comfort_penalty = (
            self.calculate_comfort_penalty(
                baseline_setpoints,
                occupancy,
                df["adaptive_comfort_target_c"].values
            )
        )

        timing["baseline_eval_sec"] = round(
            time.perf_counter() - t0, 3
        )

        # ----------------------------------------------------
        # Generate candidate pool for every day
        #
        # Each day's pool (discrete templates + Powell polish +
        # DE global search) is independent of every other day, so
        # when parallel_days is enabled this runs across a thread
        # pool instead of one day at a time. That parallelism is
        # what makes it affordable to run the heavier de_maxiter /
        # de_popsize / de_restarts settings needed to reliably
        # clear a 10%+ monthly savings target.
        # ----------------------------------------------------

        t0 = time.perf_counter()

        day_groups = list(
            df.groupby(df["timestamp"].dt.date, sort=True)
        )

        def _build_one_day(date, day_df):
            print(f"building day {date}...", flush=True)
            day_df = day_df.copy()
            candidates = self.build_day_candidate_pool(day_df)

            for candidate in candidates:
                candidate["_hours"] = (
                    day_df["hour"].values.copy()
                )
                candidate["_occupancy"] = (
                    day_df["occupancy_factor"].values.copy()
                )

            return str(date), day_df, candidates

        results_by_date = {}

        if self.parallel_days and len(day_groups) > 1:

            with ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as pool:

                futures = {
                    pool.submit(_build_one_day, date, day_df): date
                    for date, day_df in day_groups
                }

                for future in as_completed(futures):
                    date_str, day_df, candidates = future.result()
                    results_by_date[date_str] = (day_df, candidates)

        else:

            for date, day_df in day_groups:
                date_str, day_df, candidates = _build_one_day(
                    date, day_df
                )
                results_by_date[date_str] = (day_df, candidates)

        # Recover chronological order regardless of thread
        # completion order.
        day_candidates = []
        day_dates = []
        day_dfs = []

        for date, _ in day_groups:
            date_str = str(date)
            day_df, candidates = results_by_date[date_str]
            day_candidates.append(candidates)
            day_dates.append(date_str)
            day_dfs.append(day_df)

        timing["day_candidate_pools_sec"] = round(
            time.perf_counter() - t0, 3
        )

        # ----------------------------------------------------
        # GLOBAL MONTHLY OPTIMIZATION
        # ----------------------------------------------------

        t0 = time.perf_counter()

        selected_indices, optimized_plan = (
            self.global_monthly_optimizer(
                day_candidates
            )
        )

        timing["global_coordinate_descent_sec"] = round(
            time.perf_counter() - t0, 3
        )

        # ----------------------------------------------------
        # DEMAND-CHARGE-FOCUSED PEAK REFINEMENT
        #
        # The coordinate descent above picks the best whole-day
        # schedule per day from the candidate pool, but the
        # monthly demand charge is set by a SINGLE hour across
        # the entire month. This stage explicitly retargets
        # whichever day currently sets that peak and searches
        # (continuously) for a schedule that lowers it further,
        # then re-checks whether that actually helps the
        # monthly bill. It repeats, since knocking down one
        # day's peak can reveal a new "runner-up" day.
        # ----------------------------------------------------

        t0 = time.perf_counter()

        if self.enable_continuous_search:

            selected_indices, optimized_plan = (
                self.refine_monthly_peak(
                    day_candidates,
                    day_dfs,
                    selected_indices,
                    optimized_plan
                )
            )

        timing["peak_refinement_sec"] = round(
            time.perf_counter() - t0, 3
        )

        optimized_setpoints = (
            optimized_plan["setpoints"]
        )

        optimized_loads = optimized_plan["loads"]

        optimized_cost = self.calculate_monthly_cost(
            optimized_loads,
            hours
        )

        optimized_comfort_penalty = (
            optimized_plan["comfort_penalty"]
        )

        # ----------------------------------------------------
        # Savings
        # ----------------------------------------------------

        savings = (
            baseline_cost["total_cost"]
            - optimized_cost["total_cost"]
        )

        savings_pct = (
            savings /
            baseline_cost["total_cost"] *
            100
        )

        peak_reduction = (
            baseline_cost["peak_kw"]
            - optimized_cost["peak_kw"]
        )

        timing["total_sec"] = round(
            time.perf_counter() - t_total_start, 3
        )

        # ----------------------------------------------------
        # Daily strategies
        # ----------------------------------------------------

        daily_results = []

        for day_index, candidates in enumerate(
            day_candidates
        ):

            selected = candidates[
                selected_indices[day_index]
            ]

            # Determine which specialist agrees with the
            # selected global strategy.
            energy_choice = self.optimize_energy_strategy(
                candidates
            )["name"]

            peak_choice = self.optimize_peak_strategy(
                candidates
            )["name"]

            comfort_choice = self.optimize_comfort_strategy(
                candidates
            )["name"]

            daily_results.append({
                "date": day_dates[day_index],
                "strategy": selected["name"],
                "daily_energy_cost_usd": round(
                    selected["energy_cost"],
                    2
                ),
                "daily_peak_kw": round(
                    selected["peak_kw"],
                    2
                ),
                "comfort_penalty": round(
                    selected["comfort_penalty"],
                    2
                ),
                "energy_optimizer_choice": energy_choice,
                "peak_optimizer_choice": peak_choice,
                "comfort_optimizer_choice": comfort_choice
            })

        # ----------------------------------------------------
        # Hourly schedule
        # ----------------------------------------------------

        hourly_schedule = df[
            [
                "timestamp",
                "hour",
                "temperature_c",
                "occupancy_factor"
            ]
        ].copy()

        hourly_schedule[
            "baseline_setpoint_c"
        ] = baseline_setpoints

        hourly_schedule[
            "optimized_setpoint_c"
        ] = optimized_setpoints

        hourly_schedule[
            "baseline_load_kw"
        ] = np.round(
            baseline_loads,
            2
        )

        hourly_schedule[
            "optimized_load_kw"
        ] = np.round(
            optimized_loads,
            2
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {

            "summary": {

                "baseline_energy_cost_usd":
                    round(
                        baseline_cost["energy_cost"],
                        2
                    ),

                "optimized_energy_cost_usd":
                    round(
                        optimized_cost["energy_cost"],
                        2
                    ),

                "baseline_demand_charge_usd":
                    round(
                        baseline_cost["demand_charge"],
                        2
                    ),

                "optimized_demand_charge_usd":
                    round(
                        optimized_cost["demand_charge"],
                        2
                    ),

                "baseline_total_cost_usd":
                    round(
                        baseline_cost["total_cost"],
                        2
                    ),

                "optimized_total_cost_usd":
                    round(
                        optimized_cost["total_cost"],
                        2
                    ),

                "savings_usd":
                    round(savings, 2),

                "savings_pct":
                    round(savings_pct, 2),

                "baseline_peak_kw":
                    baseline_cost["peak_kw"],

                "optimized_peak_kw":
                    optimized_cost["peak_kw"],

                "peak_reduction_kw":
                    round(peak_reduction, 2),

                "baseline_comfort_penalty":
                    round(
                        baseline_comfort_penalty,
                        2
                    ),

                "optimized_comfort_penalty":
                    round(
                        optimized_comfort_penalty,
                        2
                    ),

                "global_objective":
                    round(
                        optimized_plan["objective"],
                        2
                    ),

                "timing_sec":
                    timing
            },

            "daily_strategies":
                pd.DataFrame(daily_results),

            "hourly_schedule":
                hourly_schedule
        }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)

    optimizer = HVACOptimizer()

    results = optimizer.optimize_month(df)

    summary = results["summary"]

    print("=" * 65)
    print("CoolLoad AI — Multi-Objective Monthly Optimization")
    print("=" * 65)

    print(
        f"Baseline Energy Cost       : "
        f"${summary['baseline_energy_cost_usd']:.2f}"
    )

    print(
        f"Optimized Energy Cost      : "
        f"${summary['optimized_energy_cost_usd']:.2f}"
    )

    print()

    print(
        f"Baseline Demand Charge     : "
        f"${summary['baseline_demand_charge_usd']:.2f}"
    )

    print(
        f"Optimized Demand Charge    : "
        f"${summary['optimized_demand_charge_usd']:.2f}"
    )

    print()

    print(
        f"Baseline Total Cost        : "
        f"${summary['baseline_total_cost_usd']:.2f}"
    )

    print(
        f"Optimized Total Cost       : "
        f"${summary['optimized_total_cost_usd']:.2f}"
    )

    print(
        f"Monthly Savings            : "
        f"${summary['savings_usd']:.2f} "
        f"({summary['savings_pct']:.2f}%)"
    )

    print()

    print(
        f"Baseline Monthly Peak      : "
        f"{summary['baseline_peak_kw']:.2f} kW"
    )

    print(
        f"Optimized Monthly Peak     : "
        f"{summary['optimized_peak_kw']:.2f} kW"
    )

    print(
        f"Peak Reduction             : "
        f"{summary['peak_reduction_kw']:.2f} kW"
    )

    print()

    print(
        f"Baseline Comfort Penalty   : "
        f"{summary['baseline_comfort_penalty']:.2f}"
    )

    print(
        f"Optimized Comfort Penalty  : "
        f"{summary['optimized_comfort_penalty']:.2f}"
    )

    print(
        f"Global Optimization Score : "
        f"{summary['global_objective']:.2f}"
    )

    print("=" * 65)

    timing = summary["timing_sec"]

    print("\nRuntime Breakdown:")
    print(
        f"  Baseline evaluation        : "
        f"{timing['baseline_eval_sec']:.2f}s"
    )
    print(
        f"  Day candidate pools        : "
        f"{timing['day_candidate_pools_sec']:.2f}s"
        f"  (discrete grid + Powell polish + DE, "
        f"{'parallel across days' if optimizer.parallel_days else 'sequential'})"
    )
    print(
        f"  Global coordinate descent  : "
        f"{timing['global_coordinate_descent_sec']:.2f}s"
    )
    print(
        f"  Peak/demand-charge refine  : "
        f"{timing['peak_refinement_sec']:.2f}s"
    )
    print(
        f"  TOTAL                      : "
        f"{timing['total_sec']:.2f}s"
    )
    print("=" * 65)

    print("\nDaily Strategies:")

    print(
        results["daily_strategies"].to_string(
            index=False
        )
    )