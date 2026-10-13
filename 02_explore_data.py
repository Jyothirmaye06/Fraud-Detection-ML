
import pandas as pd

# Load the dataset
df = pd.read_csv("creditcard.csv")

# 1. Dataset dimensions
print("Dataset shape:", df.shape)

# 2. Column data types
print("\nColumn data types:")
print(df.dtypes)

# 3. Missing values in each column
print("\nMissing values:")
print(df.isnull().sum())

# 4. Duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())

# 5. Summary statistics
print("\nSummary statistics:")
print(df.describe())



# Count each transaction class
print("\nTransaction class counts:")
print(df["Class"].value_counts())

# Calculate percentages
print("\nTransaction class percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

# Compare legitimate and fraudulent transactions
print("\nClass meanings:")
print("0 = Legitimate transaction")
print("1 = Fraudulent transaction")




# Find all rows that have an exact duplicate
duplicate_mask = df.duplicated(keep=False)

duplicate_rows = df[duplicate_mask]

print("\nTotal rows involved in duplicate groups:")
print(len(duplicate_rows))

print("\nClass counts among duplicate rows:")
print(duplicate_rows["Class"].value_counts())

print("\nFirst 10 duplicate rows:")
print(duplicate_rows.head(10))




# Check whether identical transaction features have conflicting labels

features = df.drop(columns=["Class"])

duplicate_feature_mask = features.duplicated(keep=False)

duplicate_feature_rows = df[duplicate_feature_mask]

label_counts = (
    duplicate_feature_rows
    .groupby(list(features.columns), dropna=False)["Class"]
    .nunique()
)

conflicting_groups = label_counts[label_counts > 1]

print("\nNumber of feature groups with conflicting labels:")
print(len(conflicting_groups))





# Explore transaction amounts
print("\nTransaction amount statistics:")
print(df["Amount"].describe())

print("\nSmallest transaction amount:")
print(df["Amount"].min())

print("\nLargest transaction amount:")
print(df["Amount"].max())

# Compare amounts by class
print("\nAverage amount by class:")
print(df.groupby("Class")["Amount"].mean())