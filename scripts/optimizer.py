from __future__ import annotations

import os
import itertools
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, differential_evolution


# ============================================================
# Electricity Tariff
# ============================================================

OFF_PEAK_RATE = 0.09        # $/kWh
ON_PEAK_RATE = 0.26         # $/kWh
DEMAND_CHARGE_RATE = 14.50  # $/kW/month

# Comfort is treated as a soft objective.
# It does NOT override the real electricity bill.
#
# IMPORTANT: this rate is a TIE-BREAKER weight, not a real dollar
# cost. At $1.00/comfort-point it was previously priced almost
# equivalently to real energy savings, which meant the optimizer
# was effectively trading away genuine bill savings to avoid a
# "cost" that customers never actually pay. Keeping it small lets
# the search chase the real electricity bill first, and only use
# comfort deviation to break ties between equally-cheap schedules.
COMFORT_PENALTY_RATE = 1e-6  # $ per comfort-point (bill tie-breaker only)


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
        "comfort_setpoint_c"
    ]

    # Hard comfort-band bounds shared by every optimizer
    # (discrete candidate search, continuous polish, DE, peak refinement).
    SETPOINT_MIN = 21.5
    SETPOINT_MAX = 25.0

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        comfort_penalty_rate: float = COMFORT_PENALTY_RATE,
        enable_continuous_search: bool = False,
        polish_top_n: int = 3,
        polish_maxiter: int = 150,
        de_maxiter: int = 40,
        de_popsize: int = 8,
        de_restarts: int = 1,
        peak_refine_rounds: int = 8,
        peak_search_weight: float = 6.0,
        random_seed: int = 42
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
            setback hours, etc.).
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
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Train model first."
            )

        self.model = joblib.load(model_path)
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

    def calculate_comfort_penalty(
        self,
        setpoints: list[float],
        occupancy: np.ndarray
    ) -> float:
        """
        Soft comfort objective.

        23C is used as the neutral comfort reference.
        Deviation matters more when occupancy is higher.

        This is NOT a medical/thermal comfort model; it is an
        optimization preference used to prevent unnecessarily
        aggressive setpoints.
        """

        setpoints = np.asarray(setpoints, dtype=float)
        occupancy = np.asarray(occupancy, dtype=float)

        deviation = np.abs(setpoints - 23.0)

        # Occupied hours matter more than unoccupied hours.
        weights = 0.25 + 0.75 * np.clip(occupancy, 0.0, 1.0)

        return float(np.sum(deviation * weights))

    # ========================================================
    # 4. Monthly Objective
    # ========================================================

    def calculate_objective(
        self,
        loads_kw: np.ndarray,
        hours: np.ndarray,
        setpoints: list[float],
        occupancy: np.ndarray
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
            occupancy
        )

        # Optimize the bill that is reported to the customer. Comfort is
        # already enforced by the hard setpoint bounds, so it can only
        # break near-equal bill ties and must not hide real savings.
        total_cost = (
            energy_cost
            + demand_charge
            + comfort_penalty * self.comfort_penalty_rate
        )

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

        features = df.copy()

        features["comfort_setpoint_c"] = setpoints

        return np.asarray(
            self.model.predict(
                features[self.FEATURE_COLS]
            ),
            dtype=float
        )

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

        # Include the best bill-first boundary schedule explicitly. The
        # continuous model has no thermal state, so a higher setpoint can
        # reduce load independently at every hour.
        schedules.append({
            "name": "Maximum Setpoint",
            "setpoints": [self.SETPOINT_MAX] * len(hours)
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

        # Hard comfort constraint
        if any(
            sp < self.SETPOINT_MIN or sp > self.SETPOINT_MAX
            for sp in setpoints
        ):
            return None

        loads = self.predict_load(
            day_df,
            setpoints
        )

        hours = day_df["hour"].values
        occupancy = day_df["occupancy_factor"].values

        metrics = self.calculate_objective(
            loads,
            hours,
            setpoints,
            occupancy
        )

        return {
            "name": schedule["name"],
            "setpoints": setpoints,
            "loads": loads,
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

        setpoints = np.clip(
            setpoints_arr,
            self.SETPOINT_MIN,
            self.SETPOINT_MAX
        )

        loads = self.predict_load(day_df, setpoints)

        metrics = self.calculate_objective(
            loads,
            hours,
            setpoints,
            occupancy
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

        bounds = Bounds(
            [self.SETPOINT_MIN] * n_hours,
            [self.SETPOINT_MAX] * n_hours
        )

        def objective(x):
            return self._day_objective_from_array(
                day_df, hours, occupancy, x,
                peak_weight=peak_weight,
                peak_cap=peak_cap
            )

        x0 = np.clip(
            np.asarray(initial_setpoints, dtype=float),
            self.SETPOINT_MIN,
            self.SETPOINT_MAX
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

        bounds = [
            (self.SETPOINT_MIN, self.SETPOINT_MAX)
            for _ in range(n_hours)
        ]

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
                    self.SETPOINT_MIN,
                    self.SETPOINT_MAX
                )
                n_pop = max(self.de_popsize * n_hours, n_hours + 4)
                rng = np.random.default_rng(restart_seed)
                jitter = rng.normal(0.0, 0.4, size=(n_pop, n_hours))
                init = np.clip(
                    base[None, :] + jitter,
                    self.SETPOINT_MIN,
                    self.SETPOINT_MAX
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

        return np.clip(
            best_x,
            self.SETPOINT_MIN,
            self.SETPOINT_MAX
        ).tolist()

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

        loads = np.asarray(all_loads, dtype=float)
        hours = np.asarray(all_hours)
        setpoints = list(all_setpoints)
        occupancy = np.asarray(all_occupancy, dtype=float)

        metrics = self.calculate_objective(
            loads,
            hours,
            setpoints,
            occupancy
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

        if df.empty:
            raise ValueError("Dataset is empty.")

        df = df.copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

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
                occupancy
            )
        )

        # ----------------------------------------------------
        # Generate candidate pool for every day
        # ----------------------------------------------------

        day_candidates = []
        day_dates = []
        day_dfs = []

        for date, day_df in df.groupby(
            df["timestamp"].dt.date,
            sort=True
        ):

            day_df = day_df.copy()

            candidates = self.build_day_candidate_pool(
                day_df
            )

            # Attach hours/occupancy metadata so the global
            # optimizer can evaluate the whole month.
            for candidate in candidates:

                candidate["_hours"] = (
                    day_df["hour"].values.copy()
                )

                candidate["_occupancy"] = (
                    day_df["occupancy_factor"].values.copy()
                )

            day_candidates.append(candidates)
            day_dates.append(str(date))
            day_dfs.append(day_df)

        # ----------------------------------------------------
        # GLOBAL MONTHLY OPTIMIZATION
        # ----------------------------------------------------

        selected_indices, optimized_plan = (
            self.global_monthly_optimizer(
                day_candidates
            )
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

        if self.enable_continuous_search:

            selected_indices, optimized_plan = (
                self.refine_monthly_peak(
                    day_candidates,
                    day_dfs,
                    selected_indices,
                    optimized_plan
                )
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
                    )
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

    print("\nDaily Strategies:")

    print(
        results["daily_strategies"].to_string(
            index=False
        )
    )