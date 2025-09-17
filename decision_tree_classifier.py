import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 1: Create a small dataset
data = {
    "Age": [25, 30, 45, 35, 40, 50, 23, 33],
    "Income": ["High", "High", "Medium", "Low", "Low", "Medium", "Low", "High"],
    "Buys_Computer": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No"]
}

df = pd.DataFrame(data)

# Step 2: Convert categorical data to numeric
df["Income"] = df["Income"].map({"Low": 0, "Medium": 1, "High": 2})
df["Buys_Computer"] = df["Buys_Computer"].map({"No": 0, "Yes": 1})

X = df[["Age", "Income"]]
y = df["Buys_Computer"]

# Step 3: Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Predictions
predictions = model.predict(X_test)

# Step 6: Results
print("Predicted values:", predictions)
print("Actual values:   ", y_test.values)
print("Model Accuracy:  ", model.score(X_test, y_test))

# Step 7: Visualize Decision Tree
plt.figure(figsize=(8,6))
plot_tree(model, feature_names=["Age", "Income"], class_names=["No", "Yes"], filled=True)
plt.show()
