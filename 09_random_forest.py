
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

df = pd.read_csv("creditcard_cleaned.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Reserve the test set for final evaluation
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split the development data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

# Create and train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on validation data only
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

from sklearn.metrics import precision_score, recall_score, f1_score

print("Validation average precision:",
      round(average_precision_score(y_val, y_val_prob), 4))

print("\nRandom Forest threshold comparison:")

for threshold in [0.5, 0.3, 0.2, 0.1]:
    y_val_pred = (y_val_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_val_pred
    ).ravel()

    precision = precision_score(
        y_val, y_val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, y_val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, y_val_pred, zero_division=0
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f} | "
        f"False alarms: {fp} | "
        f"Missed fraud: {fn}"
    )