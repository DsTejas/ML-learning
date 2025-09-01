# Employee Salary Prediction using Linear Regression
# ---------------------------------------------------
# In this project, I am predicting the salary of employees
# based on their years of experience.

import numpy as np
import matplotlib.pyplot as plt

# Dataset: Years of Experience vs Salary ($1000s)
# Created a small dataset for demonstration
experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
salary = np.array([30, 35, 40, 50, 60, 70, 85, 90, 100, 120])

# Adding bias term (for intercept)
X_b = np.c_[np.ones((len(experience), 1)), experience]

# Using Normal Equation to compute best-fit line
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(salary)

# Predicting salaries
y_pred = X_b.dot(theta_best)

# Printing model parameters
print("Intercept:", theta_best[0])
print("Slope:", theta_best[1])

# Testing with a new input (7.5 years experience)
new_exp = 7.5
predicted_salary = theta_best[0] + theta_best[1] * new_exp
print(f"Predicted Salary for {new_exp} years of experience: {predicted_salary:.2f} (in $1000s)")

# Visualization
plt.scatter(experience, salary, color="blue", label="Actual Data")
plt.plot(experience, y_pred, color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($1000s)")
plt.title("Employee Salary Prediction Model")
plt.legend()
plt.show()
