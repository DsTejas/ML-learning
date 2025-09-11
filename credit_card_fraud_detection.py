# Credit Card Fraud Detection (Classification)
# --------------------------------------------
# In this project, I am predicting if a transaction
# is fraudulent or genuine using logistic regression.

import numpy as np
import pandas as pd

# Sample dataset (Transaction Amount + Time + Location Distance)
data = {
    "Amount": [5, 200, 1500, 20, 5000, 50, 10000, 15, 70, 3000],
    "Time": [10, 120, 60, 200, 5, 600, 30, 400, 50, 25],  # minutes from last transaction
    "LocationDistance": [1, 50, 300, 2, 1000, 5, 2000, 1, 3, 500],  # km difference
    "Fraud": [0, 0, 1, 0, 1, 0, 1, 0, 0, 1]  # 1 = Fraud, 0 = Legit
}

df = pd.DataFrame(data)

# Features and target
X = df[["Amount", "Time", "LocationDistance"]].values
y = df["Fraud"].values

# Logistic Regression (scratch)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
weights = np.zeros(X.shape[1])
bias = 0
lr = 0.00001
epochs = 3000

# Training
for _ in range(epochs):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)

    # Gradients
    dw = np.dot(X.T, (y_pred - y)) / len(y)
    db = np.sum(y_pred - y) / len(y)

    # Update weights
    weights -= lr * dw
    bias -= lr * db

# Predictions
y_pred_class = (sigmoid(np.dot(X, weights) + bias) >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_pred_class == y)
print("Predictions:", list(y_pred_class))
print("Actual:", list(y))
print("Model Accuracy:", accuracy)
