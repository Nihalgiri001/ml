import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Iris.csv")

# Features and target
X = data[['SepalLengthCm', 'SepalWidthCm']] # Only first two features for visualization
y = data['Species']

# Encode labels (Species is categorical)
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction using DataFrame to match the feature names
y_pred = model.predict(X_test)

# Print the accuracy
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))