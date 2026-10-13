
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the cleaned dataset
df = pd.read_csv("creditcard_cleaned.csv")

# 2. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Create the same 60/20/20 split
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

# 4. Train Random Forest on the training set
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Get fraud probabilities on the test set
y_test_prob = model.predict_proba(X_test)[:, 1]

# 6. Apply the threshold selected on validation data
threshold = 0.3
y_test_pred = (y_test_prob >= threshold).astype(int)

# 7. Evaluate
tn, fp, fn, tp = confusion_matrix(
    y_test, y_test_pred
).ravel()

print("Final Test Results")
print("------------------")
print("Threshold:", threshold)
print("True negatives:", tn)
print("False alarms:", fp)
print("Missed fraud:", fn)
print("Fraud detected:", tp)

print("\nPrecision:", round(
    precision_score(y_test, y_test_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, y_test_pred, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, y_test_pred, zero_division=0), 4
))
print("Average precision:", round(
    average_precision_score(y_test, y_test_prob), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test, y_test_pred, zero_division=0
))