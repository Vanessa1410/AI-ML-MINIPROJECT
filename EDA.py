import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import joblib

# Load train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Check columns and handle missing values
print(train.columns)
print(test.columns)
print(train.isna().sum())

# Handle missing values by forward filling
train = train.ffill()
test = test.ffill()

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

# Initialize and train Support Vector Regression (SVR) model
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_train_svr = svr_model.predict(X_train)
y_pred_test_svr = svr_model.predict(X_test)
rmse_train_svr = np.sqrt(mean_squared_error(y_train, y_pred_train_svr))
rmse_test_svr = np.sqrt(mean_squared_error(y_test, y_pred_test_svr))

# Initialize and train Gradient Boosting Regression model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
# Calculate RMSE for Gradient Boosting model
y_pred_train_gb = gb_model.predict(X_train)
y_pred_test_gb = gb_model.predict(X_test)
rmse_train_gb = np.sqrt(mean_squared_error(y_train, y_pred_train_gb))
rmse_test_gb = np.sqrt(mean_squared_error(y_test, y_pred_test_gb))

# Initialize and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_train_lr = lr_model.predict(X_train)
y_pred_test_lr = lr_model.predict(X_test)
rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_pred_test_lr))

# Initialize and train Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
rmse_train_dt = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
rmse_test_dt = np.sqrt(mean_squared_error(y_test, y_pred_test_lr))

# Initialize and train XGBoost Regression model
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_train_xgb = xgb_model.predict(X_train)
y_pred_test_xgb = xgb_model.predict(X_test)
rmse_train_xgb = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
rmse_test_xgb = np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))

# Make predictions on training and testing sets for each model
models = [rf_model, svr_model, gb_model, lr_model, dt_model, xgb_model]
model_names = ['Random Forest', 'Support Vector', 'Gradient Boosting', 'Linear Regression', 'Decision Tree', 'XGBoost']

for model, name in zip(models, model_names):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n{name}:")
    print("Training RMSE:", rmse_train)
    print("Test RMSE:", rmse_test)
    print("R-squared score:", model.score(X_train, y_train))
    print("R-squared score:", model.score(X_test, y_test))


