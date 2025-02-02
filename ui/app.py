import streamlit as st
import requests

st.title("Microscopic Cell Classifier")

uploaded_file = st.file_uploader(
    "Upload a Microscopic Image", type=["jpg", "png", "jpeg"]
)
if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Send request to FastAPI
    response = requests.post(
        "http://localhost:8000/predict/", files={"file": open("temp.jpg", "rb")}
    )
    prediction = response.json()["prediction"]
    st.write(f"Prediction: {prediction}")
