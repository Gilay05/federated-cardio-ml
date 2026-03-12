from fastapi import FastAPI
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

app = FastAPI()

MODEL_PATH = "venv/models/main_model.pkl"
DATA_PATH = "set2_test.csv"

@app.get("/test_main_model")
def test_model():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    preds = model.predict(X)

    acc = accuracy_score(y, preds)

    return {
        "hospital": "Hospital 1",
        "test_samples": len(X),
        "accuracy": round(float(acc), 4)
    }