import os
import pickle
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature

# ===============================
# MLflow Configuration 
# ===============================
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/SuhasC-DSc/MLpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "SuhasC-DSc"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "05fe1184de8e3c9f7b17e2f1174fa1ed47b799fd"

mlflow.set_experiment("mlpipeline-experiment")

# ===============================
# Hyperparameter Tuning Function
# ===============================
def hyperparameter_tuning(X_train, y_train, params):
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=params,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search


# ===============================
# Training Function
# ===============================
def train(input_path, model_path, random_state, n_estimators, max_depth):

    # Load data
    df = pd.read_csv(input_path, header=None)
    df.columns = df.columns.astype(str)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    with mlflow.start_run():

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=random_state)

        # Infer signature
        #y_train = y_train.to_frame(name="target").astype(int)
            #Convert y_train into a DataFrame
        signature = infer_signature(X_train, y_train)

        # Hyperparameter grid
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 2, 4, 6, 8, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

        # Grid search
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        # Log best parameters
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("min_samples_split", best_model.min_samples_split)
        mlflow.log_param("min_samples_leaf", best_model.min_samples_leaf)

        # Evaluation
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # Log model to MLflow & DAGsHub registry
        mlflow.sklearn.log_model(
            best_model,
            name="model",
            registered_model_name="RandomForestClassifierModel",
            signature=signature
        )

        # Save model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Model saved to {model_path}")


# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["train"]

    train(
        input_path=params["data"],
        model_path=params["model"],
        random_state=params["random_state"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"]
    )
