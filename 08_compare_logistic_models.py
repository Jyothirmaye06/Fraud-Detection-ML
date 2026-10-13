
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

df = pd.read_csv("creditcard_cleaned.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Keep the final test set separate
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split development data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

models = {
    "Original": LogisticRegression(max_iter=1000),
    "Balanced": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
}

for name, classifier in models.items():

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    # Predict fraud probabilities on validation data
    y_val_prob = model.predict_proba(X_val)[:, 1]

    print(f"\n--- {name} Logistic Regression ---")
    print(
        "Average precision:",
        round(average_precision_score(y_val, y_val_prob), 3)
    )

    for threshold in [0.5, 0.3, 0.2, 0.1]:
        y_val_pred = (y_val_prob >= threshold).astype(int)

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
            f"F1: {f1:.3f}"
        )