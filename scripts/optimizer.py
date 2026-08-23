import os
import itertools
import joblib
import numpy as np
import pandas as pd


# ============================================================
# Electricity Tariff
# ============================================================

OFF_PEAK_RATE = 0.09        # $/kWh
ON_PEAK_RATE = 0.26         # $/kWh
DEMAND_CHARGE_RATE = 14.50  # $/kW/month

# Comfort is treated as a soft objective.
# It does NOT override the real electricity bill.
COMFORT_PENALTY_RATE = 1.00  # $ per comfort-point


MODEL_PATH = "models/hvac_forecaster.joblib"
DATA_PATH = "data/hvac_30day_dataset.csv"


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

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        comfort_penalty_rate: float = COMFORT_PENALTY_RATE
    ):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Train model first."
            )

        self.model = joblib.load(model_path)
        self.comfort_penalty_rate = comfort_penalty_rate

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

        # ----------------------------------------------------
        # Peak Shaving
        # ----------------------------------------------------

        for peak_temp in [24.0, 24.5, 25.0]:

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

        # ----------------------------------------------------
        # Pre-cooling + Peak Shaving
        # ----------------------------------------------------

        for pre_temp, peak_temp in itertools.product(
            [21.5, 22.0, 22.5],
            [24.0, 24.5, 25.0]
        ):

            schedule = []

            for h in hours:

                if 10 <= h < 12:
                    schedule.append(pre_temp)

                elif 12 <= h < 14:
                    schedule.append(22.5)

                elif 14 <= h < 19:
                    schedule.append(peak_temp)

                else:
                    schedule.append(23.0)

            schedules.append({
                "name": (
                    f"PreCool {pre_temp}C + "
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
            sp < 21.5 or sp > 25.0
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

        max_iterations = 20

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

        # ----------------------------------------------------
        # GLOBAL MONTHLY OPTIMIZATION
        # ----------------------------------------------------

        selected_indices, optimized_plan = (
            self.global_monthly_optimizer(
                day_candidates
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