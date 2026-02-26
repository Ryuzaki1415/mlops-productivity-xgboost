import shap
import matplotlib.pyplot as plt
import mlflow
import os


def run_shap_analysis(pipeline, X_train):

    model = pipeline.named_steps["model"]

    preprocessor = pipeline.named_steps["preprocessor"]

    X_processed = preprocessor.transform(X_train)

    explainer = shap.Explainer(model)

    shap_values = explainer(X_processed)

    os.makedirs("artifacts", exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_processed, show=False)

    shap_path = "artifacts/shap_summary.png"
    plt.savefig(shap_path, bbox_inches="tight")

    mlflow.log_artifact(shap_path)

    plt.close()