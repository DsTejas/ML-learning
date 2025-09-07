# Spam Email Detection (Text Classification)
# ------------------------------------------
# In this project, I am predicting whether an email is spam or not
# using a simple bag-of-words model + Naive Bayes from scratch.

import numpy as np
import pandas as pd

# Sample email dataset
data = {
    "Email": [
        "Win money now!!!", 
        "Lowest price for your meds", 
        "Hi, how are you doing?", 
        "Let's catch up tomorrow", 
        "You won a lottery!!!", 
        "Meeting at 5 pm", 
        "Cheap loans available", 
        "Are you coming to party?"
    ],
    "Label": [1, 1, 0, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Step 1: Build vocabulary
vocab = set()
for email in df["Email"]:
    for word in email.lower().split():
        vocab.add(word)
vocab = list(vocab)

# Step 2: Convert emails to feature vectors (bag-of-words)
def email_to_vector(email):
    words = email.lower().split()
    return [words.count(word) for word in vocab]

X = np.array([email_to_vector(email) for email in df["Email"]])
y = df["Label"].values

# Step 3: Naive Bayes Classifier
def train_naive_bayes(X, y):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    
    # Prior probabilities
    priors = np.zeros(n_classes)
    # Likelihoods
    likelihoods = np.zeros((n_classes, n_features))
    
    for idx, c in enumerate(classes):
        X_c = X[y == c]
        priors[idx] = X_c.shape[0] / n_samples
        likelihoods[idx, :] = (np.sum(X_c, axis=0) + 1) / (np.sum(X_c) + n_features)  # Laplace smoothing
    return priors, likelihoods

def predict_naive_bayes(X, priors, likelihoods):
    predictions = []
    for x in X:
        posteriors = []
        for idx, prior in enumerate(priors):
            posterior = np.log(prior)
            posterior += np.sum(x * np.log(likelihoods[idx]))
            posteriors.append(posterior)
        predictions.append(np.argmax(posteriors))
    return predictions

# Train model
priors, likelihoods = train_naive_bayes(X, y)

# Predictions
y_pred = predict_naive_bayes(X, priors, likelihoods)

# Accuracy
accuracy = np.mean(y_pred == y)
print("Predictions:", y_pred)
print("Actual:", list(y))
print("Model Accuracy:", accuracy)
