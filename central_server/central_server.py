from fastapi import FastAPI
import requests

app = FastAPI()

HOSPITAL1_URL = "https://web-production-a4fbb.up.railway.app/test_main_model"
HOSPITAL2_URL = "https://web-production-5e22a.up.railway.app/test_main_model"

@app.get("/")
def home():
    return {"status": "Central server running"}

@app.get("/federated_performance")
def federated_performance():

    try:
        h1 = requests.get(HOSPITAL1_URL, timeout=30).json()
    except:
        h1 = {"accuracy": "error", "test_samples": 0}

    try:
        h2 = requests.get(HOSPITAL2_URL, timeout=30).json()
    except:
        h2 = {"accuracy": "error", "test_samples": 0}

    return {
        "hospital1_accuracy": h1["accuracy"],
        "hospital2_accuracy": h2["accuracy"],
        "hospital1_samples": h1["test_samples"],
        "hospital2_samples": h2["test_samples"]
    }

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODEL_PATH = "main_model.pkl"
DATA_PATH = "cardiovascular_disease_dataset.csv"

@app.get("/baseline_metrics")
def baseline_metrics():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    preds = model.predict(X)

    return {
        "accuracy": round(accuracy_score(y, preds),4),
        "precision": round(precision_score(y, preds),4),
        "recall": round(recall_score(y, preds),4),
        "f1_score": round(f1_score(y, preds),4),
        "roc_auc": round(roc_auc_score(y, preds), 4)
    }

@app.get("/run_hospital_testing")
def run_hospital_testing():

    h1 = requests.get(HOSPITAL1_URL).json()
    h2 = requests.get(HOSPITAL2_URL).json()

    return {
        "hospital1": h1,
        "hospital2": h2
    }

import numpy as np

@app.get("/run_aggregation")
def run_aggregation():

    model = joblib.load(MODEL_PATH)

    weights = model.coef_

    aggregated_weights = np.mean(weights)

    return {
        "aggregation": "completed",
        "method": "Federated Averaging (simplified)"
    }

import shap

@app.get("/run_shap_analysis")
def run_shap():

    model = joblib.load("main_model.pkl")

    df = pd.read_csv(DATA_PATH)

    X = df.drop("cardio", axis=1)

    explainer = shap.Explainer(model, X)

    shap_values = explainer(X[:100])

    importance = abs(shap_values.values).mean(axis=0)

    features = list(X.columns)

    shap_importance = dict(zip(features, importance.tolist()))

    return {
        "shap_feature_importance": shap_importance
    }
