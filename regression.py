import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("risk_updated_dataset.csv")

# Separate features (X) and target variables (y)
X = data.drop(columns=["Risk", "Risk Level"])  # Features
y_risk = data["Risk"]  # Target variable for risk prediction
y_risk_level = data["Risk Level"]  # Target variable for risk level prediction

# Split data into training and testing sets for both risk and risk level
X_train, X_test, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
X_train, X_test, y_risk_level_train, y_risk_level_test = train_test_split(X, y_risk_level, test_size=0.2, random_state=42)

# Train Random Forest Regressor for risk prediction
rf_risk = RandomForestRegressor(n_estimators=100, random_state=42)
rf_risk.fit(X_train, y_risk_train)

# Train Random Forest Regressor for risk level prediction
rf_risk_level = RandomForestRegressor(n_estimators=100, random_state=42)
rf_risk_level.fit(X_train, y_risk_level_train)

# Predict on the testing set for both risk and risk level
y_risk_pred = rf_risk.predict(X_test)
y_risk_level_pred = rf_risk_level.predict(X_test)

# Calculate mean squared error for both predictions
mse_risk = mean_squared_error(y_risk_test, y_risk_pred)
mse_risk_level = mean_squared_error(y_risk_level_test, y_risk_level_pred)

print("Mean Squared Error for Risk Prediction:", mse_risk)
print("Mean Squared Error for Risk Level Prediction:", mse_risk_level)

import numpy as np

# Define the risk threshold
risk_threshold = 10  # Adjust this threshold according to your specific requirements

# Classify the predicted risk values into different risk levels
y_risk_pred_levels = np.where(y_risk_pred <= risk_threshold, "Low Risk", "High Risk")
y_risk_test_levels = np.where(y_risk_test <= risk_threshold, "Low Risk", "High Risk")

# Calculate the accuracy rate for risk prediction
accuracy_rate_risk = np.mean(y_risk_pred_levels == y_risk_test_levels) * 100

print("Accuracy Rate for Risk Prediction:", accuracy_rate_risk)

from sklearn.model_selection import cross_val_score

# Perform cross-validation for risk prediction
cv_scores_risk = cross_val_score(rf_risk, X, y_risk, cv=5, scoring='neg_mean_squared_error')
avg_mse_risk = -cv_scores_risk.mean()

# Perform cross-validation for risk level prediction
cv_scores_risk_level = cross_val_score(rf_risk_level, X, y_risk_level, cv=5, scoring='neg_mean_squared_error')
avg_mse_risk_level = -cv_scores_risk_level.mean()

print("Average Mean Squared Error for Risk Prediction (Cross-Validation):", avg_mse_risk)
print("Average Mean Squared Error for Risk Level Prediction (Cross-Validation):", avg_mse_risk_level)

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their ranges
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create Random Forest Regressor instance
rf = RandomForestRegressor(random_state=42)

# Create Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# Perform Grid Search on the training data
grid_search.fit(X_train, y_risk_train)

# Get the best hyperparameters and best model
best_params_risk = grid_search.best_params_
best_model_risk = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters for Risk Prediction:")
print(best_params_risk)
