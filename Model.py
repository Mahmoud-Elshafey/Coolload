import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 

DATA_PATH = "data/hvac_30day_dataset.csv"
MODEL_PATH = "models/hvac_forecaster.joblib"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please run fetch.py first.")

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "ghi_wm2",
    "hour",
    "day_of_week",
    "is_weekend",
    "occupancy_factor",
    "comfort_setpoint_c",
]
TARGET = "hvac_load_kw"

X = df[FEATURES]
y = df[TARGET]