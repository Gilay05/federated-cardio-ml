from fastapi import FastAPI
import joblib
import requests

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hospital 1 Node Running"}

@app.post("/send_model")
def send_model():
    files = {"file": open("/venv/models/hospital1_v2.pkl", "rb")}
    response = requests.post(
        "https://federated-cardio-ml-production.up.railway.app/upload_model/",
        files=files
    )
    return {"status": "model sent", "server_response": response.text}