import shap
import matplotlib.pyplot as plt
import mlflow
import os
from config import NUMERICAL_COLUMNS,DERIVED_COLUMNS,CATEGORICAL_COLUMNS

# def run_shap_analysis(pipeline, X_train):
    
#     print("Running SHAP Analysis")

#     model = pipeline.named_steps["model"]

#     preprocessor = pipeline.named_steps["preprocessor"]

#     X_processed = preprocessor.transform(X_train)

#     explainer = shap.Explainer(model)

#     shap_values = explainer(X_processed)

#     os.makedirs("artifacts", exist_ok=True)

#     plt.figure()
#     shap.summary_plot(shap_values, X_processed, show=False)

#     shap_path = "artifacts/shap_summary.png"
#     plt.savefig(shap_path, bbox_inches="tight")

#     mlflow.log_artifact(shap_path)

#     plt.close()


import shap
import matplotlib.pyplot as plt
import mlflow
import os
import pandas as pd

def run_shap_analysis(pipeline, X_train):
    
    print("Running SHAP Analysis")
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    X_processed = preprocessor.transform(X_train)

    # ── Recover feature names ──────────────────────────────────────────────────
    num_features = NUMERICAL_COLUMNS + DERIVED_COLUMNS

    cat_features = (
        preprocessor
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLUMNS)
        .tolist()
    )

    feature_names = num_features + cat_features

    # ── Wrap in DataFrame so SHAP picks up names ───────────────────────────────
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # ── SHAP ───────────────────────────────────────────────────────────────────
    explainer = shap.Explainer(model)
    shap_values = explainer(X_processed_df)

    os.makedirs("artifacts", exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_processed_df, show=False)
    
    shap_path = "artifacts/shap_summary_2.png"
    plt.savefig(shap_path, bbox_inches="tight", dpi=150)
    mlflow.log_artifact(shap_path)
    plt.close()