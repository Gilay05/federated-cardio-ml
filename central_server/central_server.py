from fastapi import FastAPI, UploadFile, File
import shutil

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Federated Learning Central Server Running"}


@app.post("/upload_model/")
async def upload_model(file: UploadFile = File(...)):
    save_path = f"models/{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"{file.filename} uploaded successfully"}

from fastapi import FastAPI
import requests

app = FastAPI()

HOSPITAL1_URL = "https://hospital1-production.up.railway.app/test_main_model"
HOSPITAL2_URL = "https://hospital2-production.up.railway.app/test_main_model"


@app.get("/performance")
def get_performance():

    try:
        h1 = requests.get(HOSPITAL1_URL).json()
    except:
        h1 = {"accuracy": "offline"}

    try:
        h2 = requests.get(HOSPITAL2_URL).json()
    except:
        h2 = {"accuracy": "offline"}

    return {
        "hospital1_accuracy": h1["accuracy"],
        "hospital2_accuracy": h2["accuracy"]
    }