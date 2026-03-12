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

    h1 = requests.get(HOSPITAL1_URL).json()
    h2 = requests.get(HOSPITAL2_URL).json()

    return {
        "hospital1_accuracy": h1["accuracy"],
        "hospital2_accuracy": h2["accuracy"],
        "hospital1_samples": h1["test_samples"],
        "hospital2_samples": h2["test_samples"]
    }