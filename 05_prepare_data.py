
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("creditcard_cleaned.csv")

# Separate input features from the target
X = df.drop(columns=["Class"])
y = df["Class"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nFraud counts in training data:")
print(y_train.value_counts())

print("\nFraud counts in testing data:")
print(y_test.value_counts())

print("\nTraining fraud percentage:")
print(round(y_train.mean() * 100, 4), "%")

print("Testing fraud percentage:")
print(round(y_test.mean() * 100, 4), "%")