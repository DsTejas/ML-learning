# House Price Prediction using Linear Regression

# In this project, I am predicting house prices
# based on the size (in square feet).

import numpy as np
import matplotlib.pyplot as plt

# Dataset: House Size (sqft) vs Price ($1000s)
# I made a small dataset for practice
house_size = np.array([650, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500]).reshape(-1, 1)
house_price = np.array([180, 200, 230, 250, 300, 330, 360, 390, 420])

# Preparing the data (adding bias term for intercept)
X_b = np.c_[np.ones((len(house_size), 1)), house_size]

# Normal Equation to calculate best fit line
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(house_price)

# Predicting house prices using the model
y_pred = X_b.dot(theta_best)

# Printing model parameters
print("Intercept:", theta_best[0])
print("Slope:", theta_best[1])

# Testing prediction for a new house of size 1600 sqft
new_size = 1600
predicted_price = theta_best[0] + theta_best[1] * new_size
print(f"Predicted Price for {new_size} sqft house: {predicted_price:.2f} (in $1000s)")

# Visualization
plt.scatter(house_size, house_price, color="blue", label="Actual Data")
plt.plot(house_size, y_pred, color="red", label="Regression Line")
plt.xlabel("House Size (sqft)")
plt.ylabel("House Price ($1000s)")
plt.title("House Price Prediction Model")
plt.legend()
plt.show()
