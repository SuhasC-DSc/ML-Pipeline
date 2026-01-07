import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/SuhasC-DSc/MLpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "SuhasC-DSc"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "05fe1184de8e3c9f7b17e2f1174fa1ed47b799fd"

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(model_path, test_data_path):
    data= pd.read_csv(test_data_path, header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    ## Load model from MLflow
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log evaluation metrics to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"Model Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate(
        model_path=params["model"],
        test_data_path=params["data"]
    )