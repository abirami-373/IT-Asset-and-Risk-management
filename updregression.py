import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv("risk_updated_dataset.csv")

# Separate features (X) and target variables (y)
X = data.drop(columns=["Risk", "Risk Level"])  # Features
y_risk = data["Risk"]  # Target variable for risk prediction
y_risk_level = data["Risk Level"]  # Target variable for risk level prediction

# Split data into training and testing sets for both risk and risk level
X_train, X_test, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
# Convert X_test and y_risk_test to pandas DataFrames
X_test_df = pd.DataFrame(X_test)
y_risk_test_df = pd.DataFrame(y_risk_test)

# Define the file paths for saving the CSV files
X_test_file_path = "X_test.csv"
y_risk_test_file_path = "y_risk_test.csv"

# Save X_test and y_risk_test to CSV files
X_test_df.to_csv(X_test_file_path, index=False)
y_risk_test_df.to_csv(y_risk_test_file_path, index=False)

print("X_test saved to:", X_test_file_path)
print("y_risk_test saved to:", y_risk_test_file_path)
X_train, X_test, y_risk_level_train, y_risk_level_test = train_test_split(X, y_risk_level, test_size=0.2, random_state=42)

# Define hyperparameters for Grid Search
param_grid = {
    'n_estimators': [50, 100],  # Reduced number of estimators
    'max_depth': [10, 20],  # Limiting the maximum depth of each tree
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize Random Forest Regressor
rf_risk = RandomForestRegressor(random_state=42)

# Perform Grid Search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_risk, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_risk_train)

# Get the best hyperparameters
best_params_risk = grid_search.best_params_

print("Best Hyperparameters for Risk Prediction:")
print(best_params_risk)

# Train Random Forest Regressor with the best hyperparameters
best_rf_risk = RandomForestRegressor(**best_params_risk, random_state=42)
best_rf_risk.fit(X_train, y_risk_train)

# Predict risk values on the test set
y_risk_pred = best_rf_risk.predict(X_test)

#predicted risk in csv
test= pd.read_csv("X_test.csv")
DF=pd.DataFrame({
    'predicted_risk':y_risk_pred,
    'Cybersecurity_Score':test['Cybersecurity_Score'],
    'Vendor_Risk_Score':test['Vendor_Risk_Score'],
    'Age':test['Age'],
    'Maintenance_Cost':test['Maintenance_Cost']
})
DF.to_csv('riskpredicted.csv')

#plotting the risk
plott= pd.read_csv("riskpredicted.csv")
x =plott['Cybersecurity_Score']
y = plott['predicted_risk']
plt.plot(x, y)
plt.xlabel('Cybersecurity_score')
plt.ylabel('Related risk')
plt.title('Rick by csvv score')
plt.grid(True)
plt.show()


# Define the risk threshold for classifying into risk levels
risk_threshold = 10  # Adjust this threshold according to your specific requirements

# Classify the predicted risk values into different risk levels
y_risk_pred_levels = np.where(y_risk_pred <= risk_threshold, "Low Risk", "High Risk")
y_risk_test_levels = np.where(y_risk_test <= risk_threshold, "Low Risk", "High Risk")



# Calculate the accuracy rate for risk prediction
accuracy_rate_risk = np.mean(y_risk_pred_levels == y_risk_test_levels) * 100

print("Accuracy Rate for Risk Prediction:", accuracy_rate_risk)

"""
# Load the new dataset for out-of-sample testing
new_data = pd.read_csv("out_risk_updated_dataset.csv")

# Separate features (X_new) and target variables (y_new_risk and y_new_risk_level) from the new dataset
X_new = new_data.drop(columns=["Risk", "Risk Level"])
y_new_risk = new_data["Risk"]
y_new_risk_level = new_data["Risk Level"]

# Use the trained model to make predictions on the new data
y_new_risk_pred = best_rf_risk.predict(X_new)

# Define the risk threshold for classifying into risk levels
risk_threshold = 10  # Adjust this threshold according to your specific requirements

# Classify the predicted risk values into different risk levels
y_new_risk_pred_levels = np.where(y_new_risk_pred <= risk_threshold, "Low Risk", "High Risk")
y_new_risk_levels = np.where(y_new_risk <= risk_threshold, "Low Risk", "High Risk")

# Calculate the accuracy rate for risk prediction on the new data
accuracy_rate_new_risk = np.mean(y_new_risk_pred_levels == y_new_risk_levels) * 100

print("Accuracy Rate for Risk Prediction on New Data:", accuracy_rate_new_risk)"""
