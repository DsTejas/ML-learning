# Student Score Prediction using Linear Regression

# In this project, I am building a simple ML model
# that predicts student exam scores based on study hours.

import numpy as np
import matplotlib.pyplot as plt

# Dataset: (Hours studied vs Scores)
# I have created a small dataset for demonstration
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
scores = np.array([10, 20, 30, 40, 50, 60, 70, 80, 85, 95])

# Preparing data by adding a bias term (for intercept)
X_b = np.c_[np.ones((len(hours), 1)), hours]

# Applying Normal Equation to calculate best-fit line parameters
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(scores)

# Making predictions using the model
y_pred = X_b.dot(theta_best)

# Printing model parameters (intercept and slope)
print("Intercept:", theta_best[0])
print("Slope:", theta_best[1])

# Testing the model with a new value
new_hours = 7.5
predicted_score = theta_best[0] + theta_best[1] * new_hours
print(f"Predicted Score for {new_hours} hours of study:", predicted_score)

# Visualizing dataset and regression line
plt.scatter(hours, scores, color="blue", label="Actual Data")  # scatter plot of data
plt.plot(hours, y_pred, color="red", label="Regression Line") # regression line
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Student Score Prediction Model")
plt.legend()
plt.show()
