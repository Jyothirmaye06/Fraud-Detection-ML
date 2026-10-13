
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("creditcard_cleaned.csv")

# Calculate skewness
amount_skewness = df["Amount"].skew()

print("Amount skewness:", round(amount_skewness, 3))

# Plot the amount distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Amount", bins=100)

plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()





# Compare amount distributions by class
print("\nAmount statistics by class:")
print(
    df.groupby("Class")["Amount"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

# Visualize amounts by class
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount",
    showfliers=False
)

plt.title("Transaction Amounts by Class")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.show()