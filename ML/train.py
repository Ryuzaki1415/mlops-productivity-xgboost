import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, GridSearchCV

from utils.config import DATA_PATH,TARGET_COLUMN,TEST_SIZE,RANDOM_STATE,MODEL_PATH,MODEL_DIR,PARAM_GRID,MLFLOW_EXPERIMENT_NAME

from feature_engineering import create_features
from pipeline import build_pipeline
from evaluate import evaluate_model
from utils.shap_analysis import run_shap_analysis
from utils.config import MLFLOW_TRACKING_URI



def train():
    print("Commencing Training.....")
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_ARTIFACT_URI"] = MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():

        print("Loading data...")

        df = pd.read_csv(DATA_PATH)
        
        

        df = create_features(df)
    

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        print("dataframe after feature engineering : ")
        
        print(df.head(10))

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

        run_shap_analysis(best_model, X_train)

        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(best_model, MODEL_PATH)

        mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="productivity_model"
)

        # Force verify artifact was logged
        active_run = mlflow.active_run()
        print(f"Run ID: {active_run.info.run_id}")
        print(f"Artifact URI: {active_run.info.artifact_uri}")
        print("Model registered to MLflow as 'productivity_model'")
        return best_model