# app.py - Loan Amount Prediction Web App
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Loan Amount Predictor",
    page_icon="💰",
    layout="centered"
)

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('loan_amount_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError as e:
    st.error("Error loading model files")
    st.info("Please ensure 'loan_amount_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# App title
st.title("Loan Amount Predictor")
st.markdown("Enter applicant details to predict the loan amount")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=None,
        placeholder="Enter applicant income"
    )

with col2:
    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=None,
        placeholder="Enter coapplicant income"
    )

loan_term = st.number_input(
    "Loan Term (days)",
    min_value=0.0,
    value=None,
    placeholder="Enter loan term"
)

credit_history = st.selectbox(
    "Credit History",
    ["Select", "Good (1)", "Bad (0)"],
    index=0
)

# Categorical inputs
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Select", "Male", "Female"], index=0)

with col2:
    married = st.selectbox("Married", ["Select", "Yes", "No"], index=0)

with col3:
    education = st.selectbox("Education", ["Select", "Graduate", "Not Graduate"], index=0)

col1, col2, col3 = st.columns(3)

with col1:
    self_employed = st.selectbox("Self Employed", ["Select", "Yes", "No"], index=0)

with col2:
    dependents = st.selectbox("Dependents", ["Select", "0", "1", "2", "3+"], index=0)

with col3:
    property_area = st.selectbox("Property Area", ["Select", "Urban", "Semiurban", "Rural"], index=0)

st.divider()

# Validation
if applicant_income is None or applicant_income <= 0:
    st.warning("Please enter a valid Applicant Income")
    st.stop()

if coapplicant_income is None or coapplicant_income < 0:
    st.warning("Please enter a valid Coapplicant Income")
    st.stop()

if loan_term is None or loan_term <= 0:
    st.warning("Please enter a valid Loan Term")
    st.stop()

if credit_history == "Select":
    st.warning("Please select Credit History")
    st.stop()

if gender == "Select":
    st.warning("Please select Gender")
    st.stop()

if married == "Select":
    st.warning("Please select Married status")
    st.stop()

if education == "Select":
    st.warning("Please select Education")
    st.stop()

if self_employed == "Select":
    st.warning("Please select Self Employed status")
    st.stop()

if dependents == "Select":
    st.warning("Please select Dependents")
    st.stop()

if property_area == "Select":
    st.warning("Please select Property Area")
    st.stop()

# Prepare input for model
total_income = applicant_income + coapplicant_income

# Create dummy variables (matching training encoding)
gender_male = 1 if gender == "Male" else 0
married_yes = 1 if married == "Yes" else 0
dependents_1 = 1 if dependents == "1" else 0
dependents_2 = 1 if dependents == "2" else 0
dependents_3 = 1 if dependents == "3+" else 0
education_graduate = 1 if education == "Graduate" else 0
self_employed_yes = 1 if self_employed == "Yes" else 0
property_area_Semiurban = 1 if property_area == "Semiurban" else 0
property_area_Urban = 1 if property_area == "Urban" else 0
credit_history_value = 1 if credit_history == "Good (1)" else 0

# Create feature array
features = np.array([[
    applicant_income,
    coapplicant_income,
    loan_term,
    credit_history_value,
    total_income,
    gender_male,
    married_yes,
    dependents_1,
    dependents_2,
    dependents_3,
    education_graduate,
    self_employed_yes,
    property_area_Semiurban,
    property_area_Urban
]])

# Scale features
features_scaled = scaler.transform(features)

# Predict button
if st.button("Predict Loan Amount", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction..."):
        prediction = model.predict(features_scaled)[0]
        
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("Predicted Loan Amount", f"${prediction:,.2f}")
        
        st.progress(min(prediction / 50000, 1.0))
        
        st.info(f"Based on the provided information, the recommended loan amount is ${prediction:,.2f}")

st.divider()
st.caption("""
Disclaimer: This is a predictive model. Final loan decisions should consider additional factors 
and be made by qualified professionals.
""")