
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

print("Loading dataset...")

df = pd.read_csv("creditcard_cleaned.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

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

fraud_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

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

print("Training XGBoost...")
model.fit(X_train, y_train)

artifact = {
    "model": model,
    "threshold": 0.9,
    "features": list(X.columns)
}

joblib.dump(artifact, "xgboost_fraud_model.joblib")

print("XGBoost model saved successfully!")
print("File: xgboost_fraud_model.joblib")
print("Selected threshold: 0.9")