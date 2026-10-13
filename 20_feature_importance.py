
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "xgboost_fraud_model.joblib"
OUTPUT_CSV = BASE_DIR / "xgboost_feature_importance.csv"
OUTPUT_PNG = BASE_DIR / "xgboost_feature_importance.png"

# Load the saved XGBoost model
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

# Get the model's feature importance scores
importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# Save all importance scores
importance_df.to_csv(OUTPUT_CSV, index=False)

# Display the top 15 features
top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Features for Fraud Detection")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

print(f"\nFull results saved to: {OUTPUT_CSV}")
print(f"Chart saved to: {OUTPUT_PNG}")