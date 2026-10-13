
import joblib
import pandas as pd

# 1. Load the saved model
artifact = joblib.load("fraud_detection_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

# 2. Select one transaction from the dataset for demonstration
df = pd.read_csv("creditcard_cleaned.csv")

# Choose a transaction row
transaction = df.drop(columns=["Class"]).iloc[[0]]

# Ensure feature columns are in the expected order
transaction = transaction[features]

# 3. Predict fraud probability
fraud_probability = model.predict_proba(transaction)[0, 1]

# 4. Apply the selected threshold
prediction = int(fraud_probability >= threshold)

# 5. Display the result
print("Transaction Fraud Detection")
print("---------------------------")
print(f"Fraud probability: {fraud_probability:.4%}")
print(f"Decision threshold: {threshold}")

if prediction == 1:
    print("Prediction: Potential fraud")
else:
    print("Prediction: Likely legitimate")