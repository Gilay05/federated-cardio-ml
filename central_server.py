from fastapi import FastAPI, UploadFile, File
import shutil
from fastapi.responses import HTMLResponse

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

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():

    html = """
    <html>
    <head>
        <title>Federated Learning Dashboard</title>
    </head>
    <body>

    <h1>Federated Learning Cardiovascular Prediction</h1>

    <h2>System Architecture</h2>
    <p>Central Server: Render</p>
    <p>Hospital Nodes: Railway</p>

    <h2>Model Performance</h2>

    <ul>
        <li>Main Model Accuracy: 0.73</li>
        <li>Hospital 1 Model Accuracy: 0.75</li>
        <li>Hospital 2 Model Accuracy: 0.76</li>
        <li>Federated Model Accuracy: 0.78</li>
    </ul>

    <h2>Federated Workflow</h2>

    <p>
    Main Model → Hospital Training → Model Aggregation → Final Model
    </p>

    </body>
    </html>
    """

    return html