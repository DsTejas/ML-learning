# Customer Churn Prediction (Classification)
# -------------------------------------------
# In this project, I am predicting whether a customer
# will leave a service (churn) or continue.

import numpy as np
import pandas as pd

# Sample dataset (Telecom-style)
data = {
    "MonthlyCharges": [20, 35, 40, 70, 90, 25, 55, 80, 95, 60],
    "TenureMonths":   [1, 12, 24, 5, 3, 18, 10, 2, 1, 15],
    "SupportCalls":   [5, 1, 0, 4, 6, 2, 3, 7, 8, 2],
    "Churn":          [1, 0, 0, 1, 1, 0, 0, 1, 1, 0]  # 1 = Left, 0 = Stayed
}

df = pd.DataFrame(data)

# Features and target
X = df[["MonthlyCharges", "TenureMonths", "SupportCalls"]].values
y = df["Churn"].values

# Logistic Regression from scratch
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.001
epochs = 2000

# Training loop
for _ in range(epochs):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    
    # Gradients
    dw = np.dot(X.T, (y_pred - y)) / len(y)
    db = np.sum(y_pred - y) / len(y)
    
    # Update
    weights -= learning_rate * dw
    bias -= learning_rate * db

# Prediction
y_pred_class = (sigmoid(np.dot(X, weights) + bias) >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_pred_class == y)
print("Predictions:", list(y_pred_class))
print("Actual:", list(y))
print("Model Accuracy:", accuracy)
