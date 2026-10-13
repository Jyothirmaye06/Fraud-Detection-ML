
import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# Load saved model and threshold
artifact = joblib.load("fraud_detection_model.joblib")
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

# Load the dataset and select 5 examples from each class
df = pd.read_csv("creditcard_cleaned.csv")

legitimate = df[df["Class"] == 0].head(5)
fraud = df[df["Class"] == 1].head(5)

demo = pd.concat([legitimate, fraud]).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Separate the known labels from the model inputs
X_demo = demo[features]
y_actual = demo["Class"]

# Predict fraud probabilities
probabilities = model.predict_proba(X_demo)[:, 1]
y_predicted = (probabilities >= threshold).astype(int)

# Display results
results = pd.DataFrame({
    "Actual": y_actual.map({
        0: "Legitimate",
        1: "Fraud"
    }),
    "Fraud_Probability": probabilities,
    "Predicted": pd.Series(y_predicted).map({
        0: "Likely legitimate",
        1: "Potential fraud"
    })
})

print("\nDemo transaction results:")
print(results.to_string(index=False))

tn, fp, fn, tp = confusion_matrix(
    y_actual,
    y_predicted,
    labels=[0, 1]
).ravel()

print("\nConfusion matrix counts:")
print("Correctly classified legitimate:", tn)
print("False alarms:", fp)
print("Missed fraud:", fn)
print("Fraud detected:", tp)

print("\nDemo metrics:")
print("Precision:", round(
    precision_score(y_actual, y_predicted, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_actual, y_predicted, zero_division=0), 3
))
print("F1-score:", round(
    f1_score(y_actual, y_predicted, zero_division=0), 3
))

print("\nClassification report:")
print(classification_report(
    y_actual,
    y_predicted,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))