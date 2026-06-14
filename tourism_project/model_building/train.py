# Production training script: loads HF data, tunes model, logs to MLflow, uploads to HF
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import joblib
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-pipeline")

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "LeonKennedy007/tourism-package-data"
MODEL_REPO   = "LeonKennedy007/tourism-package-model"
MODEL_FILENAME = "best_tourism_model_v1.joblib"

# Load train/test splits from Hugging Face
X_train = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
X_test  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
y_train = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
y_test  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()
print("Data loaded successfully.")

numeric_features = X_train.columns.tolist()
preprocessor = make_column_transformer((StandardScaler(), numeric_features))

# Gradient Boosting pipeline
gb_param_grid = {
    "gradientboostingclassifier__n_estimators": [100, 200],
    "gradientboostingclassifier__learning_rate": [0.05, 0.1],
    "gradientboostingclassifier__max_depth": [3, 5],
}
gb_pipeline = make_pipeline(preprocessor, GradientBoostingClassifier(random_state=42))

with mlflow.start_run():
    gs = GridSearchCV(gb_pipeline, gb_param_grid, cv=3, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)

    # Log all trial params
    for i, params in enumerate(gs.cv_results_["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", gs.cv_results_["mean_test_score"][i])

    best_model = gs.best_estimator_
    preds = best_model.predict(X_test)

    mlflow.log_params(gs.best_params_)
    mlflow.log_metrics({
        "test_accuracy": accuracy_score(y_test, preds),
        "test_f1": f1_score(y_test, preds),
        "test_roc_auc": roc_auc_score(y_test, preds),
    })

    # Save model and log artifact
    joblib.dump(best_model, MODEL_FILENAME)
    mlflow.log_artifact(MODEL_FILENAME, artifact_path="model")
    print("Model trained and logged.")

# Register best model to Hugging Face
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=MODEL_FILENAME,
    path_in_repo=MODEL_FILENAME,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Best model uploaded to {MODEL_REPO}")
