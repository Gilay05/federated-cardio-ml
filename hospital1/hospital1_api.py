from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict

app = FastAPI(title="Hospital 1 API")

# File paths (these are relative to the hospital1 folder)
BASE_MODEL = "hospital1/main_model.pkl"       # copy of central main model (baseline)
TRAIN_DATA = "hospital1/set2_train.csv"       # local training data (12,500)
TEST_DATA = "hospital1/set2_test.csv"         # local test data (3,000)

# -------------------------
# Utility: safe metrics
# -------------------------
def safe_metrics(model, X, y) -> Dict:
    preds = model.predict(X)
    out = {}
    try:
        out["accuracy"] = round(float(accuracy_score(y, preds)), 4)
    except Exception:
        out["accuracy"] = None
    try:
        out["precision"] = round(float(precision_score(y, preds)), 4)
    except Exception:
        out["precision"] = None
    try:
        out["recall"] = round(float(recall_score(y, preds)), 4)
    except Exception:
        out["recall"] = None
    try:
        out["f1"] = round(float(f1_score(y, preds)), 4)
    except Exception:
        out["f1"] = None
    try:
        out["roc_auc"] = round(float(roc_auc_score(y, preds)), 4)
    except Exception:
        out["roc_auc"] = None
    return out

# -------------------------
# Root
# -------------------------
@app.get("/")
def home():
    return {"status": "Hospital 1 API running"}

# -------------------------
# Test baseline model on 'samples' rows from test set (default 1500)
# Query parameter: ?samples=1500
# -------------------------
@app.get("/test_main_model")
def test_main_model(samples: int = 1500):
    try:
        model = joblib.load(BASE_MODEL)
    except Exception as e:
        return {"error": f"failed to load baseline model: {e}"}

    df = pd.read_csv(TEST_DATA)
    df = df.head(min(samples, len(df)))

    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)

    metrics = safe_metrics(model, X, y)
    metrics["samples"] = len(df)
    return metrics

# -------------------------
# Local retraining to produce hospital1_v2.pkl (uses full TRAIN_DATA)
# Endpoint: /train_local_model
# -------------------------
@app.get("/train_local_model")
def train_local_model():
    df = pd.read_csv(TRAIN_DATA)
    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    joblib.dump(model, "hospital1_v2.pkl")
    return {"status": "hospital1_v2 trained", "training_samples": len(df)}

# -------------------------
# Provide model weights to central server (weight-only exchange)
# Endpoint: /get_weights
# -------------------------
@app.get("/get_weights")
def get_weights():
    try:
        model = joblib.load("hospital1_v2.pkl")
    except Exception as e:
        return {"error": f"hospital1_v2 not found: {e}"}

    return {
        "coef": np.array(model.coef_).tolist(),
        "intercept": np.array(model.intercept_).tolist()
    }

# -------------------------
# Receive updated global model weights (central posts JSON payload)
# Endpoint: /update_global_model  (POST JSON {"coef": [...], "intercept": [...]})
# -------------------------
@app.post("/update_global_model")
def update_global_model(weights: Dict):
    try:
        model = joblib.load("hospital1_v2.pkl")
    except Exception:
        # If hospital1_v2 doesn't exist, create a fresh small model (fit on train) first
        df = pd.read_csv(TRAIN_DATA)
        X = df.drop(columns=["cardio"])
        y = df["cardio"].astype(int)
        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)

    model.coef_ = np.array(weights["coef"])
    model.intercept_ = np.array(weights["intercept"])

    joblib.dump(model, "global_model.pkl")
    return {"status": "global_model_received"}

# -------------------------
# Test the received global model on the full test set (3,000)
# Endpoint: /test_global_model
# -------------------------
@app.get("/test_global_model")
def test_global_model():
    try:
        model = joblib.load("global_model.pkl")
    except Exception as e:
        return {"error": f"global_model not found: {e}"}

    df = pd.read_csv(TEST_DATA)
    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)

    metrics = safe_metrics(model, X, y)
    metrics["samples"] = len(df)
    return metrics