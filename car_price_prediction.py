# Car Price Prediction using Linear Regression

# In this project, I am predicting the resale price of cars
# based on the number of years used.

import numpy as np
import matplotlib.pyplot as plt

# Dataset: Car Age (in years) vs Price (in $1000s)
# I created a small dataset to practice ML
car_age = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
car_price = np.array([40, 38, 35, 32, 28, 25, 22, 19, 15])

# Adding bias term (intercept column)
X_b = np.c_[np.ones((len(car_age), 1)), car_age]

# Applying Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(car_price)

# Making predictions
y_pred = X_b.dot(theta_best)

# Printing parameters
print("Intercept:", theta_best[0])
print("Slope:", theta_best[1])

# Predicting the price of a car that is 5.5 years old
new_age = 5.5
predicted_price = theta_best[0] + theta_best[1] * new_age
print(f"Predicted Price for a {new_age}-year-old car: {predicted_price:.2f} (in $1000s)")

# Visualization
plt.scatter(car_age, car_price, color="blue", label="Actual Data")
plt.plot(car_age, y_pred, color="red", label="Regression Line")
plt.xlabel("Car Age (years)")
plt.ylabel("Car Price ($1000s)")
plt.title("Car Price Prediction Model")
plt.legend()
plt.show()
