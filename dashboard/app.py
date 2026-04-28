import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FraudShield")
st.subheader("Credit Card Fraud Detection Dashboard")

st.write(
    """
    FraudShield is a machine learning dashboard that predicts whether a transaction 
    is likely to be fraudulent using an XGBoost classifier trained on the Kaggle 
    Credit Card Fraud dataset.
    """
)

# Load model and scaler
model = joblib.load("models/xgboost_fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.divider()

st.sidebar.header("Transaction Input")

st.sidebar.write("Enter transaction values below:")

# Input fields
time = st.sidebar.number_input("Time", value=0.0)
amount = st.sidebar.number_input("Amount", value=0.0)

v_features = []
for i in range(1, 29):
    value = st.sidebar.number_input(f"V{i}", value=0.0)
    v_features.append(value)

# Dataset order: Time, V1-V28, Amount
input_data = [time] + v_features + [amount]

input_df = pd.DataFrame(
    [input_data],
    columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    st.dataframe(input_df)

with col2:
    st.subheader("Prediction Result")

    if st.button("Predict Fraud Risk"):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        st.metric("Fraud Probability", f"{probability:.2%}")

        if prediction == 1:
            st.error("⚠️ High Risk: Fraudulent Transaction Detected")
        else:
            st.success("✅ Low Risk: Transaction Appears Normal")

st.divider()

st.subheader("Model Information")

st.write(
    """
    **Model Used:** XGBoost Classifier  
    **Fraud Class F1 Score:** 0.78  
    **Fraud Class Recall:** 0.87  
    **Fraud Class Precision:** 0.71  
    """
)

st.info(
    "Note: This dashboard is for portfolio and educational purposes. "
    "It is not intended for real financial decision-making."
)