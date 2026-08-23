"""
يوسّع الداتا الموجودة (data/hvac_30day_dataset.csv) بحيث كل ساعة تتكرر
بعدة قيم مختلفة لـ comfort_setpoint_c، بدل قيمة 23.0 الثابتة.

مهم: الطقس (temperature_c, humidity_pct, ghi_wm2) بتاع كل ساعة زي ما هو -
مفيش أي اتصال جديد بـ FortyGuard هنا، إحنا بس بنعيد حساب المعادلة الفيزيائية
لنفس الطقس عند setpoints مختلفة. الهدف: الموديل يتعلم إن تغيير الـ setpoint
فعلاً بيأثر على الحمل، عشان الـ optimizer يقدر يسأله سؤال حقيقي.

شغّله بعد ما يكون عندك data/hvac_30day_dataset.csv جاهز، وقبل تدريب الموديل.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "Coolload" / "data"
INPUT_PATH = DATA_DIR / "hvac_30day_dataset.csv"
OUTPUT_PATH = DATA_DIR / "hvac_30day_dataset_augmented.csv"

# القيم اللي هيختار الـ optimizer من بينها فعليًا (زوّد/قلّل حسب احتياجك)
SETPOINT_OPTIONS = [21.0, 22.0, 23.0, 24.0, 25.0]


def recompute_hvac_load(temp_c: float, humidity_pct: float, ghi_wm2: float,
                         occupancy_factor: float, comfort_setpoint_c: float) -> float:
    """نفس معادلة fetch.py بالظبط، لكن بـ setpoint متغيّر بدل الثابت."""
    sensible_load = 1.2 * max(0.0, temp_c - comfort_setpoint_c)
    latent_load = 0.5 * (humidity_pct / 100.0)
    solar_load = 0.003 * ghi_wm2
    occ_load = 0.8 * occupancy_factor
    return 15.0 + sensible_load + latent_load + solar_load + occ_load


def augment_dataset() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"الداتا الأصلية: {len(df)} صف")

    augmented_rows = []
    for _, row in df.iterrows():
        for setpoint in SETPOINT_OPTIONS:
            new_row = row.to_dict()
            new_row["comfort_setpoint_c"] = setpoint
            new_row["hvac_load_kw"] = round(
                recompute_hvac_load(
                    temp_c=row["temperature_c"],
                    humidity_pct=row["humidity_pct"],
                    ghi_wm2=row["ghi_wm2"],
                    occupancy_factor=row["occupancy_factor"],
                    comfort_setpoint_c=setpoint,
                ),
                2,
            )
            augmented_rows.append(new_row)

    df_augmented = pd.DataFrame(augmented_rows)
    df_augmented.to_csv(OUTPUT_PATH, index=False)
    print(f"الداتا الموسّعة: {len(df_augmented)} صف -> اتحفظت في {OUTPUT_PATH}")


if __name__ == "__main__":
    augment_dataset()