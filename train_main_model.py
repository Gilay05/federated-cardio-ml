# train_main_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from scipy.stats import expon, uniform

RND = 42

# Paths
TRAIN_CSV = "central_server/set1_train.csv"
TEST_CSV = "central_server/set1_test.csv"
MODEL_OUT = "central_server/main_model.pkl"   # central_server expects model here

# Load data
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

X_train = train.drop(columns=["cardio"])
y_train = train["cardio"].astype(int)

X_test = test.drop(columns=["cardio"])
y_test = test["cardio"].astype(int)

# Pipeline: scaler + logistic regression
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver="saga", max_iter=5000, random_state=RND))
])

# Hyperparameter distributions for RandomizedSearchCV
param_distributions = {
    "clf__C": expon(scale=1.0),           # inverse of regularization strength
    "clf__penalty": ["l2", "l1"],         # saga supports l1 and l2
    "clf__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_distributions,
    n_iter=40,                 # tune budget — change smaller/larger as needed
    scoring="roc_auc",         # optimize AUC for robust classification
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=RND
)

print("Starting hyperparameter search (this may take a few minutes)...")
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
best_model = search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

print("Test metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# Ensure output directory exists
out_dir = os.path.dirname(MODEL_OUT)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

# Save model (pickle)
joblib.dump(best_model, MODEL_OUT)
print("Saved best model to:", MODEL_OUT)