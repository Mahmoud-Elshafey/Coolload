"""Build the dynamic HVAC training dataset.

Run:
    python scripts/augment_data.py

Generates complete hourly control trajectories, including a dense set of
deterministic pre-cool / peak-shave templates that mirror the exact shapes
optimizer.py searches over (see thermal_model.make_dynamic_training_rows),
so the trained model has real, labeled coverage in the region the optimizer
will actually evaluate - not just flat baselines.
"""

from pathlib import Path
import pandas as pd
from thermal_model import make_dynamic_training_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "hvac_30day_dataset.csv"
OUTPUT_PATH = DATA_DIR / "hvac_30day_dataset_augmented.csv"


def augment_dataset():
    df = pd.read_csv(INPUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    dynamic = make_dynamic_training_rows(
        df,
        schedules_per_day=60,
        seed=42,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dynamic.to_csv(OUTPUT_PATH, index=False)

    print(f"Original weather rows : {len(df)}")
    print(f"Dynamic training rows : {len(dynamic)}")
    print(f"Saved                 : {OUTPUT_PATH}")


if __name__ == "__main__":
    augment_dataset()