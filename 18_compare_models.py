
import pandas as pd

# Test-split results already obtained from our earlier evaluations
results = [
    {
        "Model": "Random Forest",
        "Threshold": 0.3,
        "Average_Precision": 0.7915,
        "Precision": 0.9114,
        "Recall": 0.7579,
        "F1_Score": 0.8276,
        "False_Alarms": 7,
        "Missed_Fraud": 23,
        "Fraud_Detected": 72
    },
    {
        "Model": "XGBoost",
        "Threshold": 0.9,
        "Average_Precision": 0.8159,
        "Precision": 0.9474,
        "Recall": 0.7579,
        "F1_Score": 0.8421,
        "False_Alarms": 4,
        "Missed_Fraud": 23,
        "Fraud_Detected": 72
    }
]

comparison = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv("model_comparison.csv", index=False)

print("\nComparison saved to model_comparison.csv")

print("\n========== CURRENT RESULT ==========")
print("XGBoost has the higher F1-score and Average Precision.")
print("Both models detected 72 fraud cases and missed 23.")
print("XGBoost produced 4 false alarms; Random Forest produced 7.")