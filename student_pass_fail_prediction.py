# Student Pass/Fail Prediction using a Simple Classification Model

# In this project, I am predicting whether a student will pass or fail
# based on their study hours using a simple threshold model.

import numpy as np
import matplotlib.pyplot as plt

# Dataset: Hours studied vs Result (1 = Pass, 0 = Fail)
# I created a small dataset for practice
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Simple rule-based model: if hours > threshold, predict Pass
threshold = 5
predictions = (hours > threshold).astype(int)

# Print predictions
print("Study Hours:", hours)
print("Actual Result:", result)
print("Predicted Result:", predictions)

# Accuracy check
accuracy = np.mean(predictions == result)
print("Model Accuracy:", accuracy)

# Visualization
plt.scatter(hours, result, color="blue", label="Actual Data")
plt.plot(hours, predictions, color="red", label="Predicted Result")
plt.xlabel("Study Hours")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Student Pass/Fail Prediction Model")
plt.legend()
plt.show()
