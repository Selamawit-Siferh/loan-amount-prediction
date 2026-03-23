# app.py - Loan Amount Prediction Web App
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Loan Amount Predictor",
    layout="centered"
)

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('loan_amount_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    numerical_cols = joblib.load('numerical_cols.pkl')
    return model, scaler, feature_columns, numerical_cols

try:
    model, scaler, feature_columns, numerical_cols = load_artifacts()
except FileNotFoundError as e:
    st.error("Error loading model files")
    st.info("Please ensure all .pkl files are in the same directory.")
    st.stop()

# App title
st.title("Loan Amount Predictor")
st.markdown("Enter applicant details to predict the loan amount")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input(
        "Applicant Income (ETB)",
        min_value=0.0,
        value=None,
        placeholder="Enter applicant income in Birr"
    )

with col2:
    coapplicant_income = st.number_input(
        "Coapplicant Income (ETB)",
        min_value=0.0,
        value=None,
        placeholder="Enter coapplicant income in Birr"
    )

loan_term = st.number_input(
    "Loan Term (days)",
    min_value=0.0,
    value=None,
    placeholder="Enter loan term (180-360 days)"
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
valid_input = True
error_messages = []

if applicant_income is None or applicant_income <= 0:
    error_messages.append("Please enter a valid Applicant Income")
    valid_input = False

if coapplicant_income is None or coapplicant_income < 0:
    error_messages.append("Please enter a valid Coapplicant Income")
    valid_input = False

if loan_term is None or loan_term <= 0:
    error_messages.append("Please enter a valid Loan Term")
    valid_input = False

if credit_history == "Select":
    error_messages.append("Please select Credit History")
    valid_input = False

if gender == "Select":
    error_messages.append("Please select Gender")
    valid_input = False

if married == "Select":
    error_messages.append("Please select Married status")
    valid_input = False

if education == "Select":
    error_messages.append("Please select Education")
    valid_input = False

if self_employed == "Select":
    error_messages.append("Please select Self Employed status")
    valid_input = False

if dependents == "Select":
    error_messages.append("Please select Dependents")
    valid_input = False

if property_area == "Select":
    error_messages.append("Please select Property Area")
    valid_input = False

# Display errors
for msg in error_messages:
    st.warning(msg)

# Button
predict_button = st.button("Predict Loan Amount", type="primary", use_container_width=True)

# Process prediction
if predict_button:
    if not valid_input:
        st.error("Please fix the errors above before predicting.")
    else:
        try:
            # Calculate total income
            total_income = applicant_income + coapplicant_income
            
            # Credit history value
            credit_history_value = 1 if credit_history == "Good (1)" else 0

            # Create all 14 features
            feature_dict = {
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history_value,
                'TotalIncome': total_income,
                'Gender_Male': 1 if gender == "Male" else 0,
                'Married_Yes': 1 if married == "Yes" else 0,
                'Dependents_1': 1 if dependents == "1" else 0,
                'Dependents_2': 1 if dependents == "2" else 0,
                'Dependents_3+': 1 if dependents == "3+" else 0,
                'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
                'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
                'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
                'Property_Area_Urban': 1 if property_area == "Urban" else 0
            }

            # Create DataFrame
            input_df = pd.DataFrame([feature_dict])[feature_columns]

            # Scale ONLY the numerical columns
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

            with st.spinner("Calculating prediction..."):
                prediction = model.predict(input_df)[0]

            st.divider()

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric("Predicted Loan Amount", f"ETB {prediction:,.2f}")
                st.progress(min(prediction / 500000, 1.0))
                st.info(f"Based on the provided information, the recommended loan amount is ETB {prediction:,.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            
            with st.expander("Debug Info"):
                st.write("Numerical columns:", numerical_cols)
                st.write("All columns:", feature_columns)

st.divider()
st.caption("""
Disclaimer: This is a predictive model. Final loan decisions should consider additional factors 
and be made by qualified professionals.
""")