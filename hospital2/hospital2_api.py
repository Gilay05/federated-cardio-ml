from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict

app = FastAPI(title="Hospital 2 API")

# Central server
CENTRAL_SERVER = "https://federated-central-server.onrender.com"

# Local dataset paths
TRAIN_DATA = "hospital2/set3_train.csv"
TEST_DATA = "hospital2/set3_test.csv"

LOCAL_MODEL = "hospital2_v2.pkl"
GLOBAL_MODEL = "global_model.pkl"


# -------------------------
# Safe metrics
# -------------------------
def safe_metrics(model, X, y) -> Dict:
    preds = model.predict(X)
    out = {}

    try:
        out["accuracy"] = round(float(accuracy_score(y, preds)), 4)
    except:
        out["accuracy"] = None

    try:
        out["precision"] = round(float(precision_score(y, preds)), 4)
    except:
        out["precision"] = None

    try:
        out["recall"] = round(float(recall_score(y, preds)), 4)
    except:
        out["recall"] = None

    try:
        out["f1"] = round(float(f1_score(y, preds)), 4)
    except:
        out["f1"] = None

    try:
        out["roc_auc"] = round(float(roc_auc_score(y, preds)), 4)
    except:
        out["roc_auc"] = None

    return out


# -------------------------
# Root
# -------------------------
@app.get("/")
def home():
    return {"status": "Hospital 2 API running"}


# -------------------------
# Download base model
# -------------------------
def download_main_model():

    url = f"{CENTRAL_SERVER}/get_main_model"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to download main model")

    with open("main_model.pkl", "wb") as f:
        f.write(response.content)

    return joblib.load("main_model.pkl")


# -------------------------
# Test baseline model
# -------------------------
@app.get("/test_main_model")
def test_main_model(samples: int = 3000):

    try:
        model = download_main_model()
    except Exception as e:
        return {"error": str(e)}

    df = pd.read_csv(TEST_DATA)
    df = df.head(min(samples, len(df)))

    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)

    metrics = safe_metrics(model, X, y)
    metrics["samples"] = len(df)

    return metrics


# -------------------------
# Train local model
# -------------------------
@app.get("/train_local_model")
def train_local_model():

    df = pd.read_csv(TRAIN_DATA)

    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    joblib.dump(model, LOCAL_MODEL)

    return {
        "status": "hospital2_v2 trained",
        "training_samples": len(df)
    }


# -------------------------
# Provide weights to central
# -------------------------
@app.get("/get_weights")
def get_weights():

    try:
        model = joblib.load(LOCAL_MODEL)
    except:
        return {"error": "hospital2_v2 not found"}

    return {
        "coef": np.array(model.coef_).tolist(),
        "intercept": np.array(model.intercept_).tolist()
    }


# -------------------------
# Receive global model
# -------------------------
@app.post("/update_global_model")
def update_global_model(weights: Dict):

    df = pd.read_csv(TRAIN_DATA)

    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    model.coef_ = np.array(weights["coef"])
    model.intercept_ = np.array(weights["intercept"])

    joblib.dump(model, GLOBAL_MODEL)

    return {"status": "global_model_received"}


# -------------------------
# Test global model
# -------------------------
@app.get("/test_global_model")
def test_global_model():

    try:
        model = joblib.load(GLOBAL_MODEL)
    except Exception as e:
        return {"error": f"global_model not found: {e}"}

    df = pd.read_csv(TEST_DATA)

    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)

    metrics = safe_metrics(model, X, y)
    metrics["samples"] = len(df)

    return metrics