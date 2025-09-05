# Iris Flower Classification (Simple ML Model)
# --------------------------------------------
# In this project, I am classifying iris flowers
# into Setosa, Versicolor, or Virginica species
# based on petal and sepal dimensions.

import numpy as np
import pandas as pd

# Small version of the Iris dataset
data = {
    "SepalLength": [5.1, 7.0, 6.3, 5.8, 6.7, 5.0, 6.4, 6.9, 5.5],
    "SepalWidth":  [3.5, 3.2, 3.3, 2.7, 3.1, 3.6, 3.2, 3.1, 2.3],
    "PetalLength": [1.4, 4.7, 6.0, 5.1, 5.6, 1.4, 4.5, 5.4, 4.0],
    "PetalWidth":  [0.2, 1.4, 2.5, 1.9, 2.4, 0.2, 1.5, 2.1, 1.3],
    "Species":     ["Setosa", "Versicolor", "Virginica", "Virginica", 
                    "Virginica", "Setosa", "Versicolor", "Virginica", "Versicolor"]
}

df = pd.DataFrame(data)

# Encode species into numbers (Setosa=0, Versicolor=1, Virginica=2)
species_map = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
df["Species"] = df["Species"].map(species_map)

# Features (X) and labels (y)
X = df[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].values
y = df["Species"].values

# Simple k-NN implementation from scratch
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def predict(X_train, y_train, x_test, k=3):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    return max(set(k_nearest_labels), key=k_nearest_labels.count)

# Predict for each data point
predictions = [predict(X, y, x, k=3) for x in X]

# Accuracy
accuracy = np.mean(predictions == y)
print("Predictions:", predictions)
print("Actual:", list(y))
print("Model Accuracy:", accuracy)
