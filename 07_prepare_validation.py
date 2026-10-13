
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("creditcard_cleaned.csv")

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# First split: 80% development data, 20% final test data
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: divide the 80% development data
# into 60% training and 20% validation of the full dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

print("\nFraud cases in each set:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Testing:", int(y_test.sum()))

print("\nFraud percentage in each set:")
print("Training:", round(y_train.mean() * 100, 4), "%")
print("Validation:", round(y_val.mean() * 100, 4), "%")
print("Testing:", round(y_test.mean() * 100, 4), "%")