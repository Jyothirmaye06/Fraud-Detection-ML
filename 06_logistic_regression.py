
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# 1. Load cleaned data
df = pd.read_csv("creditcard_cleaned.csv")

# 2. Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Create a pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
))
])

# 5. Train using only training data
model.fit(X_train, y_train)

# 6. Predict labels for the test data
y_pred = model.predict(X_test)

# 7. Evaluate
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Evaluate precision-recall performance
y_prob = model.predict_proba(X_test)[:, 1]

print(
    "\nAverage precision (PR metric):",
    round(average_precision_score(y_test, y_prob), 4)
)


from sklearn.metrics import precision_score, recall_score, f1_score

print("\nThreshold comparison:")

for threshold in [0.5, 0.3, 0.2, 0.1]:
    y_threshold_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test, y_threshold_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_threshold_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_threshold_pred, zero_division=0
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )