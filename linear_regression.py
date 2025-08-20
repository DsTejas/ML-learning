import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Create a small dataset
data = {
    "Area": [1000, 1500, 2000, 2500, 3000],
    "Price": [100000, 150000, 200000, 250000, 300000]
}
df = pd.DataFrame(data)

# Step 2: Split data into features (X) and target (y)
X = df[["Area"]]
y = df["Price"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test)

# Step 6: Show results
print("Test Data (Area):")
print(X_test)
print("\nPredicted Prices:")
print(predictions)
print("\nActual Prices:")
print(y_test.values)
