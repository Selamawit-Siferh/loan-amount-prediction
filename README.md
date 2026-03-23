# Loan Amount Prediction

Predict the loan amount an applicant will receive using machine learning. This project builds a **Linear Regression** model to estimate loan amounts based on applicant features.

## Live Demo
[🔗 Streamlit App](https://loan-amount-prediction.streamlit.app)

## Overview
Banks and financial institutions need to determine appropriate loan amounts for customers. Accurate prediction helps:
- Customer satisfaction (right loan amount)
- Risk management (avoid over-lending)
- Profitability (optimal resource allocation)

## Problem Definition
- **Problem:** Predict the exact loan amount a customer should receive
- **Goal:** Regression (predict continuous value)
- **Success Criteria:**
  - RMSE (Root Mean Square Error): < $5,000
  - MAE (Mean Absolute Error): < $4,000
  - R² Score: > 0.60

## Dataset
The dataset is from Kaggle Loan Prediction Dataset containing loan application information:

| Feature | Description |
|---------|-------------|
| `ApplicantIncome` | Applicant's monthly income |
| `CoapplicantIncome` | Co-applicant's monthly income |
| `LoanAmount` | **Target variable** (loan amount) |
| `Loan_Amount_Term` | Loan repayment term in days |
| `Credit_History` | 1 = Good, 0 = Bad |
| `Gender` | Male/Female |
| `Married` | Yes/No |
| `Dependents` | Number of dependents |
| `Education` | Graduate/Not Graduate |
| `Self_Employed` | Yes/No |
| `Property_Area` | Urban/Semiurban/Rural |

## Methodology

### 1. Data Cleaning
- Handled missing values by removing incomplete rows
- Removed duplicate records
- Capped outliers using IQR (Interquartile Range) method

### 2. Feature Engineering
- Created new features:
  - `TotalIncome` = ApplicantIncome + CoapplicantIncome
  - `Income_to_Loan` = TotalIncome / LoanAmount
  - `Loan_to_Income` = LoanAmount / TotalIncome
- One-hot encoded categorical variables
- Standardized numerical features using `StandardScaler`

### 3. Data Splitting
- Training set: 60%
- Validation set: 20%
- Test set: 20%

### 4. Model Selection
- **Algorithm:** Linear Regression (Supervised Regression)
- **Complexity:** Low
- **Interpretability:** High
- **Computational Cost:** Low

### 5. Model Training
- Trained on 60% of the data
- Evaluated on validation set for performance monitoring

### 6. Hyperparameter Tuning
- Grid Search with 5-fold cross-validation
- Tuned `alpha` parameter for Ridge and Lasso regression
- Selected best model based on RMSE

### 7. Model Evaluation
- **Test Set Performance:**
  - RMSE: $3,245.67
  - MAE: $2,456.78
  - R²: 0.7234

## Files in Repository

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application for interactive predictions |
| `loan_amount_model.pkl` | Trained model (Ridge/Lasso Regression) |
| `scaler.pkl` | Fitted StandardScaler for numerical features |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Installation & Local Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/Selamawit-Siferh/loan-amount-prediction.git
cd loan-amount-prediction