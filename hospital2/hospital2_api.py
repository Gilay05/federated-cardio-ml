# hospital1/hospital1_api.py
from fastapi import FastAPI
import traceback

_import_error = None
try:
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
except Exception:
    _import_error = traceback.format_exc()

app = FastAPI()

@app.get("/")
def home():
    # If import failed, return the traceback so we can see exactly why
    if _import_error:
        return {"status": "import_error", "trace": _import_error}
    return {"status": "Hospital 2 API running"}

# keep the API but only if imports succeeded
if _import_error is None:
    MODEL_PATH = "hospital2/main_model.pkl"
    DATA_PATH = "hospital2/set3_test.csv"


    @app.get("/test_main_model")
    def test_model():
        try:
            model = joblib.load(MODEL_PATH)
            df = pd.read_csv(DATA_PATH)

            X = df.drop("cardio", axis=1)
            y = df["cardio"]

            preds = model.predict(X)
            acc = accuracy_score(y, preds)

            return {
                "hospital": "Hospital 2",
                "test_samples": len(X),
                "accuracy": round(float(acc), 4)
            }

        except Exception as e:
            return {"error": str(e)}

