
import pandas as pd

# Load the original dataset
df = pd.read_csv("creditcard.csv")

print("Original dataset shape:", df.shape)

# Count extra exact duplicate rows
duplicate_count = df.duplicated().sum()
print("Extra duplicate rows:", duplicate_count)

# Remove exact duplicate rows
df_clean = df.drop_duplicates().copy()

print("Cleaned dataset shape:", df_clean.shape)
print("Rows removed:", len(df) - len(df_clean))

# Check class distribution after cleaning
print("\nClass counts after cleaning:")
print(df_clean["Class"].value_counts())

# Save a separate cleaned dataset
df_clean.to_csv("creditcard_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")