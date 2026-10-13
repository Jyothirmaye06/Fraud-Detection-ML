
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier

print("Loading dataset...")

df = pd.read_csv("creditcard_cleaned.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Split data into training, validation, and test sets
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

# Calculate class imbalance weight
fraud_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

print("Training XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=fraud_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_val)[:, 1]

ap = average_precision_score(y_val, probabilities)

print("\nValidation Average Precision:", round(ap, 4))

thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
results = []

# Evaluate every threshold
for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "False_Alarms": int(fp),
        "Missed_Fraud": int(fn),
        "True_Positives": int(tp)
    })

results_df = pd.DataFrame(results)

print("\nXGBoost Validation Threshold Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "xgboost_validation_results.csv",
    index=False
)

print("\nNumber of thresholds evaluated:", len(results_df))
print("Results saved to xgboost_validation_results.csv")