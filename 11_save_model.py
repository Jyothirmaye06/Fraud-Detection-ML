
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the cleaned dataset
df = pd.read_csv("creditcard_cleaned.csv")

# Separate input features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Reproduce the same training split
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

# Train the selected model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save the model and prediction threshold together
model_artifact = {
    "model": model,
    "threshold": 0.3,
    "features": list(X.columns)
}

joblib.dump(model_artifact, "fraud_detection_model.joblib")

print("Model saved successfully!")
print("File: fraud_detection_model.joblib")
print("Selected threshold:", model_artifact["threshold"])