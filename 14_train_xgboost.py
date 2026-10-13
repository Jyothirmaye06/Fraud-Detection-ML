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

# 1. Load dataset

print("Loading dataset...")
df = pd.read_csv("creditcard_cleaned.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# 2. Recreate the same training/validation/test split

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

print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)
print("Training fraud cases:", int(y_train.sum()))
print("Validation fraud cases:", int(y_val.sum()))

# 3. Account for class imbalance

legitimate_count = int((y_train == 0).sum())
fraud_count = int((y_train == 1).sum())
scale_weight = legitimate_count / fraud_count

print("\nScale positive weight:", round(scale_weight, 2))

# 4. Train XGBoost 

model = XGBClassifier(
n_estimators=300,
max_depth=4,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
scale_pos_weight=scale_weight,
objective="binary:logistic",
eval_metric="aucpr",
random_state=42,
n_jobs=-1,
tree_method="hist"
)

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)
print("Training completed!")

# 5. Calculate validation probabilities

probabilities = model.predict_proba(X_val)[:, 1]

ap = average_precision_score(y_val, probabilities)
print("\nValidation Average Precision:", round(ap, 4))

# 6. Evaluate all thresholds

thresholds = [0.5, 0.3, 0.2, 0.1]
results = []

print("\nXGBoost Validation Threshold Comparison")
print("-" * 75)

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

row = {
    "Threshold": threshold,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "False_Alarms": int(fp),
    "Missed_Fraud": int(fn),
    "True_Positives": int(tp)
}

results.append(row)

print(
    f"Threshold={threshold:.1f} | "
    f"Precision={precision:.3f} | "
    f"Recall={recall:.3f} | "
    f"F1={f1:.3f} | "
    f"False alarms={fp} | "
    f"Missed fraud={fn}"
)


# 7. Save and display all results

results_df = pd.DataFrame(results)
results_df.to_csv("xgboost_validation_results.csv", index=False)

print("\nComplete results:")
print(results_df.to_string(index=False))
print("\nNumber of thresholds evaluated:", len(results_df))
print("Results saved to xgboost_validation_results.csv")
