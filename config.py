import os

DATA_PATH = r"C:\Users\dheer\Downloads\Kaggle_dataset\Smartphone_Usage_Productivity_Dataset_50000.csv"

TARGET_COLUMN = "Work_Productivity_Score"

DROP_COLUMNS = ["User_ID"]

NUMERICAL_COLUMNS = [
    "Age",
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
    "phone_usage_intensity",
    "social_media_ratio",
    "weekend_usage_ratio",
    "sleep_deficit",
    "caffeine_per_hour",
    "stress_sleep_ratio",
    "screen_stress_interaction",
    "apps_per_hour"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_DIR = "models"
MODEL_NAME = "xgboost_pipeline.pkl"
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