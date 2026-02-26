import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
MODEL_PATH  = os.path.join(BASE_DIR, "artifacts", "xgboost_pipeline_v5.pkl")
print(MODEL_PATH)
DATA_PATH   = os.path.join(BASE_DIR, "data", "screen_sleep_stress_highsignal.csv")
print(DATA_PATH)

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Work_Productivity_Score"

NUMERICAL_COLUMNS = [
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
    "Stress_Level",
    "App_Usage_Count",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Occupation",
    "Device_Type"
]

DERIVED_COLUMNS = [
    "social_media_ratio",
    "weekend_usage_ratio",
    "sleep_deficit",
    "stress_sleep_ratio",
    "screen_stress_interaction",
]
# Raw inputs required from the user (includes Sleep_Hours for engineering)
RAW_INPUT_COLUMNS = NUMERICAL_COLUMNS + ["Sleep_Hours"] + CATEGORICAL_COLUMNS

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Services ───────────────────────────────────────────────────────────────────
API_HOST       = "0.0.0.0"
API_PORT       = 8000
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "ministral-3:3b"
FASTAPI_BASE_URL = "http://localhost:8000"
