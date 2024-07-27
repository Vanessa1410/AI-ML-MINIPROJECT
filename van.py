import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load train and test datasets
train = pd.read_csv('train.csv')

# Check columns and handle missing values
print(train.columns)
print(train.isna().sum())

# Handle missing values by forward filling
train = train.ffill()

# Drop 'Id' column from train dataset
train = train.drop("Id", axis=1)

# Separate categorical and numerical columns
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=np.number).columns.tolist()

# Encode categorical features using One-Hot Encoding
encoder = OneHotEncoder(drop='first', sparse_output=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train[categorical_cols]))
train_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# Replace original categorical columns with encoded ones
train = pd.concat([train.drop(categorical_cols, axis=1), train_encoded], axis=1)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'].values.reshape(-1, 1))
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
Q1 = train[numerical_cols].quantile(0.25)
Q3 = train[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Define the threshold for outliers
threshold = 1.5

# Identify outliers
outliers = ((train[numerical_cols] < (Q1 - threshold * IQR)) | (train[numerical_cols] > (Q3 + threshold * IQR))).any(axis=1)

# Remove outliers from the dataset
train = train[~outliers]


# Split train dataset into features and target variable
X = train.drop(columns='SalePrice')
y = train['SalePrice']

# Split train dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest Regression model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Calculate RMSE for Random Forest model
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)
rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))

print(rmse_test_rf)