import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Load trained model
model = xgb.XGBClassifier()
model.load_model("../models/fraud_model.json")

# Load scaler
with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("💳 Fraud Detection System")
st.write("Upload transaction data or enter manually to check for fraud.")

# Option 1: Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    scaled_data = scaler.transform(data)
    preds = model.predict(scaled_data)
    data["Prediction"] = ["Fraud" if p==1 else "Not Fraud" for p in preds]
    st.write(data.head())
    st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv")

# Option 2: Manual Input
st.subheader("Or enter transaction details manually")
V1 = st.number_input("V1", -50.0, 50.0, 0.0)
V2 = st.number_input("V2", -50.0, 50.0, 0.0)
Amount = st.number_input("Transaction Amount", 0.0, 5000.0, 10.0)

if st.button("Predict"):
    # code to process input and predict
    result = model.predict(scaled_data)
    st.write(f"Prediction: {'Fraud' if result[0]==1 else 'Legit'}")
