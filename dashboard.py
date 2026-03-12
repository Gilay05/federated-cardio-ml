import streamlit as st
import requests
import pandas as pd

st.title("Federated Learning Cardiovascular Dashboard")

SERVER_URL = "https://YOUR-RENDER-URL/performance"

data = requests.get(SERVER_URL).json()

h1 = data["hospital1_accuracy"]
h2 = data["hospital2_accuracy"]

st.metric("Hospital 1 Accuracy", h1)
st.metric("Hospital 2 Accuracy", h2)

federated = (h1 + h2) / 2

results = pd.DataFrame({
    "Model": ["Hospital1","Hospital2","Federated"],
    "Accuracy":[h1,h2,federated]
})

st.bar_chart(results.set_index("Model"))