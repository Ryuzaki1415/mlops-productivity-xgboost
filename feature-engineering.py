import pandas as pd
import numpy as np
from config import DROP_COLUMNS

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Drop unnecessary columns
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Avoid division by zero
    df["phone_usage_intensity"] = df["Daily_Phone_Hours"] / (df["App_Usage_Count"] + 1)
    df["social_media_ratio"] = df["Social_Media_Hours"] / (df["Daily_Phone_Hours"] + 0.1)
    df["weekend_usage_ratio"] = df["Weekend_Screen_Time_Hours"] / (df["Daily_Phone_Hours"] + 0.1)

    df["sleep_deficit"] = 8 - df["Sleep_Hours"]

    df["caffeine_per_hour"] = df["Caffeine_Intake_Cups"] / (df["Daily_Phone_Hours"] + 0.1)

    df["stress_sleep_ratio"] = df["Stress_Level"] / (df["Sleep_Hours"] + 0.1)

    df["screen_stress_interaction"] = df["Daily_Phone_Hours"] * df["Stress_Level"]

    df["apps_per_hour"] = df["App_Usage_Count"] / (df["Daily_Phone_Hours"] + 0.1)

    return df