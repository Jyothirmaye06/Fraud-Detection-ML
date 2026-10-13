
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "creditcard_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "demo_mixed_transactions.csv"

# Load the cleaned dataset
df = pd.read_csv(DATA_PATH)

# Select real examples from both classes
legitimate = df[df["Class"] == 0].head(5)
fraudulent = df[df["Class"] == 1].head(5)

# Combine and shuffle the 10 transactions
demo = pd.concat(
    [legitimate, fraudulent],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# Save features AND actual labels
demo.to_csv(OUTPUT_PATH, index=False)

print("Demo CSV created successfully!")
print(f"File: {OUTPUT_PATH}")
print(f"Shape: {demo.shape}")

print("\nActual class counts:")
print(demo["Class"].value_counts().rename({
    0: "Legitimate",
    1: "Fraud"
}))

print("\nFirst 5 rows:")
print(demo[["Time", "Amount", "Class"]].head())