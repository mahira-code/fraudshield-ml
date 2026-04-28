# 🛡️ ALLANAI.FraudShield — Credit Card Fraud Detection System

A machine learning-based fraud detection system designed to identify anomalous financial transactions using both supervised and unsupervised learning approaches. The project includes model training, evaluation, explainability, and a live Streamlit dashboard for real-time fraud prediction.

---

## 🚀 Overview

Fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small portion of total activity (~0.17%). This project aims to build a robust system capable of detecting fraud with high recall while maintaining precision.

---

## 📊 Dataset

- Source: Kaggle Credit Card Fraud Dataset  
- Total Records: 284,807  
- Fraud Cases: 492 (~0.17%)  
- Features:
  - `V1–V28`: PCA-transformed anonymized features  
  - `Time`, `Amount`  
  - Target: `Class` (0 = Normal, 1 = Fraud)

---

## 🧱 Project Architecture

ALLANAI.FraudShield/
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── models/
│   ├── xgboost_fraud_model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── 01_fraudshield_eda_model.ipynb
├── requirements.txt
└── README.md

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights:

- Dataset is highly imbalanced (~0.17% fraud)
- Most transactions are low-value with long-tail distribution
- Fraud patterns are not linearly separable
- PCA features require non-linear models

---

## 🤖 Models Implemented

### 1. XGBoost (Supervised Learning)

- Handles imbalanced data using `scale_pos_weight`
- Captures complex non-linear relationships

**Performance:**
- Precision: 0.71  
- Recall: 0.87  
- F1 Score: 0.78  

---

### 2. Isolation Forest (Unsupervised Learning)

- Detects anomalies without labelled data
- Useful as baseline anomaly detector

**Performance:**
- Precision: 0.29  
- Recall: 0.30  
- F1 Score: 0.30  

---

## ⚖️ Model Comparison

| Model | Precision | Recall | F1 Score |
|------|----------|--------|---------|
| XGBoost | 0.71 | 0.87 | 0.78 |
| Isolation Forest | 0.29 | 0.30 | 0.30 |

### Key Insight

Supervised learning significantly outperforms unsupervised anomaly detection in this dataset due to availability of labelled fraud data.

---

## 🔍 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to interpret model predictions by identifying the contribution of each feature.

This improves:
- Transparency  
- Trust in predictions  
- Debugging capability  

---

## 🖥️ Streamlit Dashboard

An interactive dashboard allows real-time fraud prediction.

### Features:
- Input transaction details
- Predict fraud probability
- Display risk level
- Show model performance metrics

### Run locally:

```bash
streamlit run dashboard/app.py