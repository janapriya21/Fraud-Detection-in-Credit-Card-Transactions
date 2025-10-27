import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# 1️⃣ Load Model and Scaler
# -------------------------------
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# -------------------------------
# 2️⃣ Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("### Detect fraudulent transactions using trained ML model")

st.markdown("---")

# -------------------------------
# 3️⃣ Define Feature Columns
# -------------------------------
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS.insert(0, "Time")
FEATURE_COLUMNS.append("Amount")

# -------------------------------
# 4️⃣ Input Section
# -------------------------------
st.sidebar.header("🔢 Enter Transaction Details")

def get_user_input():
    time = st.sidebar.number_input("Time (seconds)", min_value=0.0, max_value=200000.0, value=0.0, step=100.0)
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, max_value=5000.0, value=0.0, step=10.0)
    
    v_values = []
    st.sidebar.markdown("### Enter PCA Feature Values (V1–V28)")
    for i in range(1, 29):
        v = st.sidebar.number_input(f"V{i}", value=0.0, step=0.01, format="%.5f")
        v_values.append(v)
    
    user_input = {"Time": time}
    for i, v in enumerate(v_values, start=1):
        user_input[f"V{i}"] = v
    user_input["Amount"] = amount

    return user_input

user_input = get_user_input()

st.markdown("### 🧾 Transaction Summary")
st.dataframe(pd.DataFrame([user_input]))

# -------------------------------
# 5️⃣ Prediction Section
# -------------------------------
if st.button("🚀 Predict Fraud"):
    # Convert to DataFrame and scale
    user_data = pd.DataFrame([user_input], columns=FEATURE_COLUMNS)
    scaled_data = scaler.transform(user_data)

    # Predict
    prediction = model.predict(scaled_data)[0]
    prediction_proba = model.predict_proba(scaled_data)
    fraud_prob = prediction_proba[0][1]
    legit_prob = prediction_proba[0][0]

    # Results
    st.subheader("🔍 Prediction Results")
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Fraud Probability: {fraud_prob*100:.4f}%)")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {fraud_prob*100:.4f}%)")

    col1, col2 = st.columns(2)
    col1.metric(label="Fraud Probability", value=f"{fraud_prob*100:.4f}%")
    col2.metric(label="Legit Probability", value=f"{legit_prob*100:.4f}%")

    st.progress(float(fraud_prob))

st.markdown("---")
st.caption("Developed by Janapriya | Powered by XGBoost, Isolation Forest, and Streamlit 🌟")
