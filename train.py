import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    MODEL_DIR,
    PARAM_GRID,
    MLFLOW_EXPERIMENT_NAME,
)

from feature_engineering import create_features
from pipeline import build_pipeline
from evaluate import evaluate_model
from shap_analysis import run_shap_analysis


def train():

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():

        print("Loading data...")

        df = pd.read_csv(DATA_PATH)

        df = create_features(df)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print("Building pipeline...")

        pipeline = build_pipeline()

        print("Running GridSearch...")

        grid_search = GridSearchCV(
            pipeline,
            PARAM_GRID,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        print("Best Params:", grid_search.best_params_)

        mlflow.log_params(grid_search.best_params_)

        rmse, mae, r2 = evaluate_model(best_model, X_test, y_test)

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        print("Running SHAP analysis...")
        run_shap_analysis(best_model, X_train)

        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(best_model, MODEL_PATH)

        mlflow.sklearn.log_model(best_model, "model")

        print("Model saved to:", MODEL_PATH)

        return best_model