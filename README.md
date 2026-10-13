# Credit Card Fraud Detection System

## Overview
A machine learning project that identifies potentially fraudulent credit card transactions. It compares classification models and provides a Streamlit application for interactive predictions.

## Models
- Logistic Regression
- Random Forest
- XGBoost

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- SHAP

## Features
- Fraud probability predictions
- CSV upload and batch predictions
- Adjustable classification threshold
- Model evaluation metrics
- Feature importance and SHAP explanations

## Dataset
Uses a public historical credit card transaction dataset. The original data contains anonymized features and a transaction amount. The project does not connect to live banking systems.

## Run the Project
1. Create and activate a Python virtual environment.
2. Install the dependencies listed in `requirements.txt`.
3. Run the Streamlit application:

   `python -m streamlit run app.py`

## Limitations
Model performance on historical data may not represent performance on new transactions. This project is for educational and demonstration purposes, not production banking decisions.