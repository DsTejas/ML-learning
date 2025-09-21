# Basic Naive Bayes Example
# Author: Your Name
# Description: Classifying emails as Spam or Not Spam using Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Create a small text dataset
data = {
    "Email": [
        "Win a lottery now",
        "Congratulations you won prize",
        "Call me tomorrow",
        "Let’s have lunch today",
        "Earn money fast",
        "Meeting scheduled at 5 PM",
        "Free entry in contest",
        "Project deadline tomorrow"
    ],
    "Label": ["Spam", "Spam", "Not Spam", "Not Spam", "Spam", "Not Spam", "Spam", "Not Spam"]
}

df = pd.DataFrame(data)

# Step 2: Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Email"])

# Step 3: Encode target variable
df["Label"] = df["Label"].map({"Not Spam": 0, "Spam": 1})
y = df["Label"]

# Step 4: Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 5: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Predictions
predictions = model.predict(X_test)

# Step 7: Results
print("Predicted labels:", predictions)
print("Actual labels:   ", y_test.values)
print("Model Accuracy:  ", model.score(X_test, y_test))
