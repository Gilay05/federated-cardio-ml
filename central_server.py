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