import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load CSV file
df = pd.read_csv("data.csv")

# Display basic information about dataset
print(df.info())
print(df.describe())
print(df.head())

print(df.isnull().sum())
# There are no missing values in the dataset

df['month'] = pd.to_datetime(df['month'])    # Convert to date format
df['resale_price'] = df['resale_price'] / 1000    # Convert to thousands
df['remaining_lease_years'] = df['remaining_lease'].str.extract(r'(\d+)', expand=False)    # extract year

df.to_csv('cleaned_data.csv', index=False)

# EDA
# Distribution of resale prices
plt.figure(figsize=(10, 6))
sns.histplot(df['resale_price'], bins=50, kde=True, discrete=False)
plt.title('Distribution of Resale Prices')
plt.xlabel('Resale Price')
plt.ylabel('Frequency')
plt.show()

# Average resale price by town
plt.figure(figsize=(12, 8))
avg_price_by_town = df.groupby('town')['resale_price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_by_town.values, y=avg_price_by_town.index, palette='viridis')
plt.title('Average Resale Price by Town')
plt.xlabel('Average Resale Price')
plt.ylabel('Town')
plt.show()

# Resale price by flat type
plt.figure(figsize=(12, 8))
sns.boxplot(x='flat_type', y='resale_price', data=df, palette='muted')
plt.title('Resale Price by Flat Type')
plt.xlabel('Flat Type')
plt.ylabel('Resale Price')
plt.show()

# Resale price by storey range
plt.figure(figsize=(12, 8))
sns.boxplot(x='storey_range', y='resale_price', data=df, palette='muted')
plt.title('Resale Price by Storey Range')
plt.xlabel('Storey Range')
plt.ylabel('Resale Price')
plt.xticks(rotation=45)
plt.show()

# Lease commencement date and resale price
plt.figure(figsize=(12, 8))
avg_price_by_lease = df.groupby('remaining_lease_years')['resale_price'].mean().sort_values(ascending=True)
sns.barplot(x=avg_price_by_lease.index, y=avg_price_by_lease.values, palette='viridis')
plt.title('Remaining Lease Years vs Resale Price')
plt.xlabel('Remaining Lease Years')
plt.ylabel('Resale Price')
plt.show()
