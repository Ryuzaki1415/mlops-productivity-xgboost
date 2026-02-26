import os

DATA_PATH = r"D:\ML_project\dataset\screen_sleep_stress_highsignal.csv"

TARGET_COLUMN = "Work_Productivity_Score"

DROP_COLUMNS = ["User_ID"," Age",]

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

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_DIR = "models"
MODEL_NAME = "xgboost_pipeline_v4.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

MLFLOW_EXPERIMENT_NAME = "screen_productivity_xgboost"

PARAM_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}


print("Loaded Configs!")