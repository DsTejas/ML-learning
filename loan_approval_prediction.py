# Loan Approval Prediction (Classification)
# -----------------------------------------
# In this project, I am predicting whether a loan application
# will be approved based on applicant details.

import numpy as np
import pandas as pd

# Sample Loan Dataset
data = {
    "ApplicantIncome": [2500, 5000, 6000, 3500, 4000, 3000, 8000, 12000, 1500, 2200],
    "CreditScore":     [650, 720, 690, 580, 710, 600, 750, 800, 500, 640],
    "LoanAmount":      [100, 200, 250, 120, 150, 100, 300, 400, 80, 90],
    "Approved":        [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]  # 1 = Approved, 0 = Rejected
}

df = pd.DataFrame(data)

# Features and target
X = df[["ApplicantIncome", "CreditScore", "LoanAmount"]].values
y = df["Approved"].values

# Logistic Regression from scratch
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.00001   # small learning rate because values are large
epochs = 2000

# Training with Gradient Descent
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
