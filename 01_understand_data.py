
import pandas as pd

# Load the transaction dataset
df = pd.read_csv("creditcard.csv")

# Display the first five transactions
print(df.head())

# Display dataset size
print("\nDataset shape:", df.shape)

# Display all column names
print("\nColumn names:")
print(df.columns.tolist())