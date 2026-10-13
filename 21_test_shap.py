
import joblib
import shap
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_fraud_model.joblib"
DATA_PATH = BASE_DIR / "creditcard_cleaned.csv"

# Load the saved model and dataset
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

df = pd.read_csv(DATA_PATH)

# Select one legitimate and one fraudulent example
legitimate = df[df["Class"] == 0].head(1)
fraudulent = df[df["Class"] == 1].head(1)

examples = pd.concat([legitimate, fraudulent])

X_examples = examples[features]

# Explain XGBoost predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_examples)

for i in range(len(examples)):
    actual_class = int(examples.iloc[i]["Class"])
    probability = model.predict_proba(
        X_examples.iloc[[i]]
    )[0, 1]

    print("\n" + "=" * 50)
    print(f"Example {i + 1}")
    print(f"Actual class: {actual_class}")
    print(f"Fraud probability: {probability:.4%}")
    print("Top 5 features influencing the model score:")

    contributions = pd.DataFrame({
        "Feature": features,
        "SHAP_Value": shap_values.values[i],
    })

    contributions["Absolute_Impact"] = (
        contributions["SHAP_Value"].abs()
    )

    print(
        contributions
        .sort_values("Absolute_Impact", ascending=False)
        .head(5)[["Feature", "SHAP_Value"]]
        .to_string(index=False)
    )