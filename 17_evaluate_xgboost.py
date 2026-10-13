
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Loading dataset and saved XGBoost model...")

df = pd.read_csv("creditcard_cleaned.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Recreate the same split used during training
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

artifact = joblib.load("xgboost_fraud_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]

# Predict fraud probabilities on the test split
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# Calculate metrics
ap = average_precision_score(y_test, probabilities)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print("\n========== XGBOOST TEST RESULTS ==========")
print("Threshold:", threshold)
print("Average Precision:", round(ap, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("True Negatives:", tn)
print("False Alarms:", fp)
print("Missed Fraud:", fn)
print("Fraud Detected:", tp)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

# Save individual predictions for later analysis
output = X_test.copy()
output["Actual_Class"] = y_test
output["Fraud_Probability"] = probabilities
output["Predicted_Class"] = predictions

output.to_csv("xgboost_test_predictions.csv", index=False)

print("\nPredictions saved to xgboost_test_predictions.csv")