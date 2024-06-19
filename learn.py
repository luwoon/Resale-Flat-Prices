import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("cleaned_data.csv")

df['year'] = pd.to_datetime(df['month']).dt.year
df['month_of_year'] = pd.to_datetime(df['month']).dt.month
columns_to_drop = ['block', 'street_name', 'lease_commence_date', 'remaining_lease', 'month']
df = df.drop(columns_to_drop, axis=1)

# one-hot encode categorical variables
cat_cols = ['town', 'flat_type', 'storey_range', 'flat_model','year', 'month_of_year']
data_encoded = pd.get_dummies(df, columns=cat_cols)

# split data into features and target variable
X = data_encoded.drop('resale_price', axis=1)
y = data_encoded['resale_price']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize model
model = RandomForestRegressor(random_state=42)

# train model
model.fit(X_train, y_train)

# predict on test set
y_pred = model.predict(X_test)

# evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

"""
Mean Squared Error: 1335.8902159520185
Root Mean Squared Error: 36.54983195518166
R-squared: 0.9437443839206258
"""
