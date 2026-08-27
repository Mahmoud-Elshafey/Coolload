"""Train the time-dependent CoolLoad HVAC forecaster.

v2 changes (fixing a real deployed bug - see below):

- The model file previously shipped (`hvac_forecaster.joblib`) turned out to
  have been trained on only 8 features - missing `previous_setpoint_c` and
  `indoor_temp_c` entirely. optimizer.py's predict_load() still *built*
  those two columns correctly (real thermal-memory reconstruction), but the
  model silently ignored them, so every hour was predicted independently
  with zero awareness of pre-cooling or setpoint history. That is the
  actual reason the optimizer couldn't clear 10%: not a weak search, a
  stale model artifact.
- This script now (1) always validates the augmented dataset against
  thermal_model.THERMAL_MODEL_VERSION and rebuilds if it's missing or
  stale, (2) trains with monotonic constraints so the model is PHYSICALLY
  GUARANTEED to predict non-increasing load as comfort_setpoint_c rises
  (and non-decreasing load as indoor_temp_c/temperature_c rise) - this
  embeds domain knowledge directly into the model and prevents tree-noise
  from ever masking the setpoint signal again, and (3) hard-fails after
  training if the saved model's feature schema doesn't exactly match
  FEATURES, instead of silently deploying a crippled model.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "hvac_30day_dataset.csv"
AUGMENTED_DATA_PATH = DATA_DIR / "hvac_30day_dataset_augmented.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "hvac_forecaster.joblib"

FEATURES = [
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

TARGET = "hvac_load_kw"

REQUIRED_COLUMNS = FEATURES + [TARGET, "timestamp"]

# Monotonic constraints, in the SAME ORDER as FEATURES.
#   +1  -> prediction must be non-decreasing in this feature
#   -1  -> prediction must be non-increasing in this feature
#    0  -> unconstrained
#
# temperature_c      +1  hotter outside  -> never predicts LESS load
# comfort_setpoint_c -1  higher setpoint -> never predicts MORE load
# indoor_temp_c       +1  hotter indoor state -> never predicts LESS load
#
# This is what guarantees the optimizer always sees a real, exploitable
# gradient on the setpoint - it can no longer be flattened out by tree
# noise or sparse training coverage in any region of the search space.
MONOTONE_CONSTRAINTS = (1, 0, 0, 0, 0, 0, 0, -1, 0, 1)


def build_dynamic_dataset():
    """Rebuild the augmented dataset using the thermal-memory generator."""
    print("\n[INFO] Rebuilding dynamic augmented dataset...")

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from thermal_model import make_dynamic_training_rows

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    raw = pd.read_csv(RAW_DATA_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    dynamic = make_dynamic_training_rows(
        raw,
        schedules_per_day=60,
        seed=42,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dynamic.to_csv(AUGMENTED_DATA_PATH, index=False)

    print(f"[INFO] Original rows : {len(raw)}")
    print(f"[INFO] Dynamic rows  : {len(dynamic)}")
    print(f"[INFO] Saved         : {AUGMENTED_DATA_PATH}")

    return dynamic


def load_training_data():
    """Load the correct training data and rebuild if missing, incomplete, or stale.

    The staleness check (`_thermal_version`) is the fix for the exact class
    of bug that shipped last time: an augmented CSV (or a model trained on
    it) silently surviving a change to the physics/feature pipeline.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from thermal_model import THERMAL_MODEL_VERSION

    if not AUGMENTED_DATA_PATH.exists():
        return build_dynamic_dataset()

    df = pd.read_csv(AUGMENTED_DATA_PATH)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(
            "\n[WARNING] The current augmented dataset is missing required "
            f"columns: {missing}"
        )
        return build_dynamic_dataset()

    stamped_version = (
        df["_thermal_version"].iloc[0]
        if "_thermal_version" in df.columns and len(df) > 0
        else None
    )
    if stamped_version != THERMAL_MODEL_VERSION:
        print(
            f"\n[WARNING] Augmented dataset was built with thermal model "
            f"version {stamped_version!r}, current version is "
            f"{THERMAL_MODEL_VERSION!r}. Rebuilding so training data "
            f"matches the current physics."
        )
        return build_dynamic_dataset()

    return df


def main():
    df = load_training_data()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Training dataset is still missing required columns: "
            + ", ".join(missing)
        )

    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(f"Not enough valid training rows: {len(df)}")

    # Time-ordered split: do NOT randomly shuffle temporal trajectories.
    split = int(len(df) * 0.80)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print("\n" + "=" * 60)
    print("CoolLoad AI — Time-Dependent HVAC Model Training (v2)")
    print("=" * 60)
    print(f"Dataset rows : {len(df)}")
    print(f"Train rows   : {len(train_df)}")
    print(f"Test rows    : {len(test_df)}")
    print("\nFeatures (with monotonic constraints):")
    for feature, constraint in zip(FEATURES, MONOTONE_CONSTRAINTS):
        tag = {1: "↑ non-decreasing", -1: "↓ non-increasing", 0: ""}[constraint]
        print(f"  - {feature:22s} {tag}")

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=42,
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # ------------------------------------------------------------
    # HARD SCHEMA CHECK - this is the guard against exactly the bug
    # that was found: a model silently saved/deployed with fewer
    # features than the optimizer expects to feed it.
    # ------------------------------------------------------------
    booster_features = list(model.get_booster().feature_names)
    if booster_features != FEATURES:
        raise RuntimeError(
            "Refusing to save model: trained feature schema does not match "
            f"FEATURES.\n  Trained on : {booster_features}\n  Expected   : "
            f"{FEATURES}\nThis is the exact bug that shipped previously - "
            "do not override this check."
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"MAE : {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")
    print(f"R²  : {r2:.4f}")

    importances = model.feature_importances_
    print("\nFeature importances:")
    for feature, imp in sorted(
        zip(FEATURES, importances), key=lambda x: -x[1]
    ):
        print(f"  {feature:22s} {imp:.4f}")

    print(f"\n[OK] Model schema verified: {len(booster_features)} features "
          f"match FEATURES exactly.")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()