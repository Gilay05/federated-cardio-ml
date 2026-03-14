from fastapi import FastAPI
import requests
import numpy as np
import joblib
import pandas as pd
import shap
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

app = FastAPI(title="Central Federated Server")

from fastapi.responses import FileResponse

# Allow hospitals to download the base model
@app.get("/get_main_model")
def get_main_model():
    return FileResponse(
        MODEL_PATH,
        media_type="application/octet-stream",
        filename="main_model.pkl"
    )

# ----------------------------
# Configure hospital base URLs (no trailing slash)
# Replace these with YOUR deployed URLs
# ----------------------------
H1 = "https://web-production-a4fbb.up.railway.app"   # Hospital 1 base
H2 = "https://web-production-5e22a.up.railway.app"   # Hospital 2 base

# Endpoints on hospitals
H1_TEST = f"{H1}/test_main_model"
H2_TEST = f"{H2}/test_main_model"
H1_TRAIN = f"{H1}/train_local_model"
H2_TRAIN = f"{H2}/train_local_model"
H1_GET = f"{H1}/get_weights"
H2_GET = f"{H2}/get_weights"
H1_UPDATE = f"{H1}/update_global_model"
H2_UPDATE = f"{H2}/update_global_model"
H1_TEST_GLOBAL = f"{H1}/test_global_model"
H2_TEST_GLOBAL = f"{H2}/test_global_model"

# Local central paths (central_server folder is the working dir for this file on Render)
MODEL_PATH = "central_server/main_model.pkl"                # main baseline model stored in central_server folder
DATA_PATH = "cardiovascular_disease_dataset.csv"  # dataset at repo root
MAIN_V2_PATH = "main_model_v2.pkl"

# ----------------------------
# Utility: safe metrics calculation
# ----------------------------
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

# ----------------------------
# Baseline metrics (main model on full central dataset)
# ----------------------------
@app.get("/baseline_metrics")
def baseline_metrics():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)
    return safe_metrics(model, X, y)

# ----------------------------
# Hospital 1 logic: follows spec exactly
#   - test on 1500
#   - compare with baseline accuracy
#   - if reduced -> retrain on hospital training -> test full 3000
#   - else -> test 1700 (1500+200); if degraded -> retrain; else still retrain (as per your spec).
# Returns decision and final full-test metrics.
# ----------------------------
@app.get("/hospital1_logic")
def hospital1_logic():
    baseline = baseline_metrics()
    baseline_acc = baseline.get("accuracy")

    # step 1: initial test on 1500
    r1500 = requests.get(f"{H1_TEST}?samples=1500", timeout=30).json()
    a1500 = r1500.get("accuracy")

    # If we couldn't compute baseline or a1500, return error info
    if baseline_acc is None or a1500 is None:
        return {"error": "baseline or hospital initial test failed", "baseline": baseline, "initial": r1500}

    # Decision branch
    if a1500 < baseline_acc:
        # accuracy reduced -> retrain locally
        _ = requests.get(H1_TRAIN, timeout=120).json()
        full = requests.get(f"{H1_TEST}?samples=3000", timeout=60).json()
        return {"decision": "retrained_due_to_reduction", "initial": r1500, "final_full": full}
    else:
        # not reduced -> incremental +200 test
        r1700 = requests.get(f"{H1_TEST}?samples=1700", timeout=30).json()
        a1700 = r1700.get("accuracy")
        if a1700 is None:
            # if incremental test failed, still retrain per spec
            _ = requests.get(H1_TRAIN, timeout=120).json()
            full = requests.get(f"{H1_TEST}?samples=3000", timeout=60).json()
            return {"decision": "incremental_failed_then_retrained", "initial": r1500, "incremental": r1700, "final_full": full}
        # If incremental accuracy is lower than initial -> degraded -> retrain
        if a1700 < a1500:
            _ = requests.get(H1_TRAIN, timeout=120).json()
            full = requests.get(f"{H1_TEST}?samples=3000", timeout=60).json()
            return {"decision": "degraded_after_increment_retrained", "initial": r1500, "incremental": r1700, "final_full": full}
        else:
            # not degraded; per spec we still retrain then test full
            _ = requests.get(H1_TRAIN, timeout=120).json()
            full = requests.get(f"{H1_TEST}?samples=3000", timeout=60).json()
            return {"decision": "stable_after_increment_retrained", "initial": r1500, "incremental": r1700, "final_full": full}

# ----------------------------
# Hospital 2 logic (mirror of hospital1 but with 3000 and 3200 / full 4000)
# ----------------------------
@app.get("/hospital2_logic")
def hospital2_logic():
    baseline = baseline_metrics()
    baseline_acc = baseline.get("accuracy")

    r3000 = requests.get(f"{H2_TEST}?samples=3000", timeout=30).json()
    a3000 = r3000.get("accuracy")

    if baseline_acc is None or a3000 is None:
        return {"error": "baseline or hospital initial test failed", "baseline": baseline, "initial": r3000}

    if a3000 < baseline_acc:
        _ = requests.get(H2_TRAIN, timeout=180).json()
        full = requests.get(f"{H2_TEST}?samples=4000", timeout=90).json()
        return {"decision": "retrained_due_to_reduction", "initial": r3000, "final_full": full}
    else:
        r3200 = requests.get(f"{H2_TEST}?samples=3200", timeout=30).json()
        a3200 = r3200.get("accuracy")
        if a3200 is None:
            _ = requests.get(H2_TRAIN, timeout=180).json()
            full = requests.get(f"{H2_TEST}?samples=4000", timeout=90).json()
            return {"decision": "incremental_failed_then_retrained", "initial": r3000, "incremental": r3200, "final_full": full}
        if a3200 < a3000:
            _ = requests.get(H2_TRAIN, timeout=180).json()
            full = requests.get(f"{H2_TEST}?samples=4000", timeout=90).json()
            return {"decision": "degraded_after_increment_retrained", "initial": r3000, "incremental": r3200, "final_full": full}
        else:
            _ = requests.get(H2_TRAIN, timeout=180).json()
            full = requests.get(f"{H2_TEST}?samples=4000", timeout=90).json()
            return {"decision": "stable_after_increment_retrained", "initial": r3000, "incremental": r3200, "final_full": full}

# ----------------------------
# Aggregate weights (download weights only & FedAvg)
# ----------------------------
@app.get("/aggregate_models")
def aggregate_models():
    w1 = requests.get(H1_GET, timeout=30).json()
    w2 = requests.get(H2_GET, timeout=30).json()

    if "error" in w1 or "error" in w2:
        return {"error": "failed to fetch weights", "w1": w1, "w2": w2}

    coef1 = np.array(w1["coef"])
    coef2 = np.array(w2["coef"])
    int1 = np.array(w1["intercept"])
    int2 = np.array(w2["intercept"])

    avg_coef = (coef1 + coef2) / 2.0
    avg_int = (int1 + int2) / 2.0

    model = joblib.load(MODEL_PATH)
    model.coef_ = avg_coef
    model.intercept_ = avg_int

    joblib.dump(model, MAIN_V2_PATH)
    return {"status": "aggregation_complete", "model": MAIN_V2_PATH}

# ----------------------------
# Distribute the global model weights to hospitals
# ----------------------------
@app.get("/distribute_global_model")
def distribute_global_model():
    try:
        model = joblib.load(MAIN_V2_PATH)
    except Exception as e:
        return {"error": f"main_model_v2 not found: {e}"}

    payload = {"coef": np.array(model.coef_).tolist(), "intercept": np.array(model.intercept_).tolist()}

    res1 = requests.post(H1_UPDATE, json=payload, timeout=30).json()
    res2 = requests.post(H2_UPDATE, json=payload, timeout=30).json()

    return {"hospital1_update": res1, "hospital2_update": res2}

# ----------------------------
# Test the distributed global model on both hospitals
# ----------------------------
@app.get("/test_global_model")
def test_global_model():
    r1 = requests.get(H1_TEST_GLOBAL, timeout=30).json()
    r2 = requests.get(H2_TEST_GLOBAL, timeout=30).json()
    return {"hospital1_global_test": r1, "hospital2_global_test": r2}

# ----------------------------
# SHAP explainability on central aggregated model
# ----------------------------
@app.get("/run_shap_analysis")
def run_shap_analysis():
    try:
        model = joblib.load(MAIN_V2_PATH)
    except Exception as e:
        return {"error": f"main_model_v2 not found: {e}"}

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["cardio"])

    # LinearExplainer is appropriate for linear models (fast)
    try:
        explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X[:200])
        # shap_values shape depends on model type; convert to absolute mean
        import numpy as _np
        importance = _np.mean(_np.abs(shap_values), axis=0).tolist()
    except Exception:
        # fallback generic explainer (slower)
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X[:200])
        importance = np.mean(np.abs(shap_values.values), axis=0).tolist()

    features = list(X.columns)
    return {"model": "main_model_v2", "shap_feature_importance": dict(zip(features, importance))}

# ----------------------------
# Full pipeline: run both hospital logics, aggregate, distribute, test, shap
# ----------------------------
@app.get("/run_full_pipeline")
def run_full_pipeline():
    h1 = hospital1_logic()
    h2 = hospital2_logic()
    agg = aggregate_models()
    dist = distribute_global_model()
    test = test_global_model()
    shap_out = run_shap_analysis()
    return {
        "hospital1_logic": h1,
        "hospital2_logic": h2,
        "aggregation": agg,
        "distribute": dist,
        "global_test": test,
        "shap": shap_out
    }