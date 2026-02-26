import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering transformations.
    Expects raw input columns including Sleep_Hours.
    Safe for single-row inference DataFrames.
    """
    df = df.copy()

    # Guard: avoid division by zero or near-zero denominators
    phone_hours_safe = df["Daily_Phone_Hours"].replace(0, np.nan)
    sleep_hours_safe = df["Sleep_Hours"].replace(0, np.nan)
    app_count_safe   = df["App_Usage_Count"].replace(0, np.nan)

    # ── Core derived features ──────────────────────────────────────────────────
    df["sleep_deficit"] = (8 - df["Sleep_Hours"]).clip(lower=0)

    df["stress_sleep_ratio"] = (
        df["Stress_Level"] / sleep_hours_safe
    ).clip(upper=10).fillna(10)

    df["screen_stress_interaction"] = (
        df["Daily_Phone_Hours"] * df["Stress_Level"]
    )

    df["social_media_ratio"] = (
        df["Social_Media_Hours"] / phone_hours_safe
    ).clip(0, 1).fillna(0)


    df["weekend_usage_ratio"] = (
        df["Weekend_Screen_Time_Hours"] / phone_hours_safe
    ).clip(upper=5).fillna(0)

    return df
