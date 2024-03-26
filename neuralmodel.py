import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv("risk_updated_dataset.csv")

# Split the dataset into features (X) and target variables (y)
X = data.drop(columns=["Risk", "Risk Level"])
y_risk = data["Risk"]
y_risk_level = data["Risk Level"]

# Split the data into training and testing sets
X_train, X_test, y_risk_train, y_risk_test, y_risk_level_train, y_risk_level_test = train_test_split(
    X, y_risk, y_risk_level, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model to predict risk
model.fit(X_train_scaled, y_risk_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_risk_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_scaled, y_risk_test)
print("Test Accuracy for Risk Prediction:", accuracy)

# Modify the model for risk level prediction
model.add(Dense(4, activation='softmax'))  # Output layer for multiclass classification
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model to predict risk level
model.fit(X_train_scaled, y_risk_level_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_risk_level_test))

# Evaluate the model on the test set for risk level prediction
loss, accuracy = model.evaluate(X_test_scaled, y_risk_level_test)
print("Test Accuracy for Risk Level Prediction:", accuracy)
