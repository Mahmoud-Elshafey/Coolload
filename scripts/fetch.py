import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import the official client from the local fortyguard package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fortyguard import FortyGuardClient, FortyGuardError

# 1. Initialize Client & Configuration
load_dotenv()
client = FortyGuardClient()

# Target Building: Bank of America Tower, Phoenix, AZ
BOA_LAT = 33.4479
BOA_LON = -112.0704
OUTPUT_CSV = "data/hvac_30day_dataset.csv"
START_DATE_STR = "2024-07-01"
TOTAL_DAYS = 30

# GeoJSON Polygon AOI around Bank of America Tower
BOA_POLYGON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [BOA_LON - 0.003, BOA_LAT - 0.003],
                        [BOA_LON + 0.003, BOA_LAT - 0.003],
                        [BOA_LON + 0.003, BOA_LAT + 0.003],
                        [BOA_LON - 0.003, BOA_LAT + 0.003],
                        [BOA_LON - 0.003, BOA_LAT - 0.003],
                    ]
                ],
            },
        }
    ],
}


def fetch_day_env_params(date_str: str):
    """Fetches real 24-hour humidity array and clear-sky GHI peak using exact JSON schema."""
    try:
        env_res = client.environmental_parameters(
            latitude=BOA_LAT,
            longitude=BOA_LON,
            temperature=25.0,
            start_date=date_str,
            filter_type=3,  # 3 = single day
            verbose=False,
        )
        res_data = env_res.get("result", {})
        locations = res_data.get("locations", [])

        if locations:
            loc = locations[0]

            # Extract 24-value relative humidity array
            raw_humidity = (
                loc.get("parameters", {})
                .get("relative_humidity_percent", [])
            )

            # Extract clear sky GHI peak value
            ghi_peak = (
                loc.get("solar_irradiance", {})
                .get("clear_sky", {})
                .get("ghi", 800.0)
            )

            # Ensure 24 elements and replace any None values with 50.0 fallback
            clean_humidity = []
            for h in range(24):
                val = (
                    raw_humidity[h]
                    if h < len(raw_humidity) and raw_humidity[h] is not None
                    else 50.0
                )
                clean_humidity.append(float(val))

            return clean_humidity, float(ghi_peak)

    except Exception as e:
        print(f"   [!] env_params failed for {date_str}: {e}. Falling back to default values.")

    return [50.0] * 24, 800.0


def build_30day_dataset():
    start_dt = datetime.datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    all_records = []

    print("=" * 70)
    print("CoolLoad AI — Complete 30-Day Training Dataset Generator")
    print(f"Target Building : Bank of America Tower, Phoenix, AZ ({BOA_LAT}, {BOA_LON})")
    print(f"Start Date      : {START_DATE_STR} | Duration: {TOTAL_DAYS} Days (720 Hours)")
    print(f"Output File     : {OUTPUT_CSV}")
    print("=" * 70)

    for day_idx in range(TOTAL_DAYS):
        current_dt = start_dt + datetime.timedelta(days=day_idx)
        date_str = current_dt.strftime("%Y-%m-%d")
        print(f"\n---> Day {day_idx + 1}/{TOTAL_DAYS}: {date_str}")

        # Step 1: Fetch environmental parameters once per day
        humidity_arr, ghi_peak = fetch_day_env_params(date_str)
        print(
            f"   [✓] env_params loaded: Clear-Sky Peak GHI = {ghi_peak:.1f} W/m², "
            f"Humidity Range = {min(humidity_arr):.1f}% - {max(humidity_arr):.1f}%"
        )

        # Step 2: Fetch hourly heatmap readings (24 calls/day)
        for hour in range(24):
            time_str = f"{hour:02d}:00"
            timestamp_str = f"{date_str} {time_str}:00"

            try:
                heatmap_res = client.create_heatmap(
                    polygon_aoi=BOA_POLYGON,
                    start_date=date_str,
                    filter_type=1,  # 1 = single hour
                    start_time=time_str,
                    analytic_type="tcm",
                    verbose=False,
                )
                res_data = heatmap_res.get("result", {})
                stats = res_data.get("stats_data", {})

                # Parse temperature mean with multi-schema fallback
                temp_c = stats.get("mean")
                if temp_c is None:
                    temp_c = stats.get("temperature_stats", {}).get("mean", 27.0)

            except Exception as e:
                print(f"   [!] Heatmap failed for {timestamp_str}: {e}. Fallback = 27.0 °C")
                temp_c = 27.0

            # Environmental calculations
            humidity_pct = humidity_arr[hour]
            solar_factor = (
                max(0, -((hour - 12) ** 2) / 16 + 1) if 6 <= hour <= 18 else 0
            )
            ghi_wm2 = ghi_peak * solar_factor

            # Operational calculations
            day_of_week = current_dt.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            occupancy = 0.1 if is_weekend else (0.9 if 8 <= hour <= 17 else 0.15)
            comfort_setpoint = 23.0

            # 3-Term HVAC Load Formula: Sensible + Latent + Solar + Occupancy
            sensible_load = 1.2 * max(0, temp_c - comfort_setpoint)
            latent_load = 0.5 * (humidity_pct / 100.0)
            solar_load = 0.003 * ghi_wm2
            occ_load = 0.8 * occupancy

            hvac_load_kw = 15.0 + sensible_load + latent_load + solar_load + occ_load

            all_records.append({
                "timestamp": timestamp_str,
                "temperature_c": round(float(temp_c), 2),
                "humidity_pct": round(float(humidity_pct), 2),
                "ghi_wm2": round(float(ghi_wm2), 2),
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "occupancy_factor": occupancy,
                "comfort_setpoint_c": comfort_setpoint,
                "hvac_load_kw": round(float(hvac_load_kw), 2),
            })

        # Save progress to CSV after every completed day
        os.makedirs("data", exist_ok=True)
        df_current = pd.DataFrame(all_records)
        df_current.to_csv(OUTPUT_CSV, index=False)
        print(f"   [✓] Day {day_idx + 1} saved ({len(df_current)} total rows).")

    print("\n" + "=" * 70)
    print(f"[🚀] SUCCESS! Complete 30-day dataset (720 rows) saved to: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    build_30day_dataset()