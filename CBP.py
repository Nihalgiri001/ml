import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]
test1 = [[
14.2, 20.5, 90.2, 600.5, 0.10,
0.15, 0.12, 0.08, 0.18, 0.06,
0.30, 1.20, 2.50, 30.0, 0.007,
0.03, 0.04, 0.01, 0.02, 0.003,
15.5, 25.0, 100.0, 700.0, 0.14,
0.25, 0.30, 0.10, 0.25, 0.08
]]

test2 = [[
20.5, 25.3, 140.0, 1200.0, 0.12,
0.30, 0.35, 0.20, 0.25, 0.07,
0.60, 1.50, 4.00, 80.0, 0.009,
0.08, 0.10, 0.05, 0.03, 0.006,
25.0, 30.0, 180.0, 1800.0, 0.16,
0.40, 0.50, 0.25, 0.35, 0.10
]]
test3 = [[
17.0, 22.0, 110.0, 850.0, 0.11,
0.20, 0.18, 0.12, 0.20, 0.065,
0.40, 1.30, 3.00, 45.0, 0.008,
0.05, 0.06, 0.02, 0.025, 0.004,
20.0, 27.0, 140.0, 1100.0, 0.15,
0.30, 0.35, 0.15, 0.28, 0.09
]]


data = pd.read_csv(url, names=columns)

data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

data = data.drop("id", axis=1)

X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test1_df = pd.DataFrame(test1, columns=X.columns)
test2_df = pd.DataFrame(test2, columns=X.columns)
test3_df = pd.DataFrame(test3, columns=X.columns)

test1_scaled = scaler.transform(test1_df)
test2_scaled = scaler.transform(test2_df)
test3_scaled = scaler.transform(test3_df)

print("Test 1 Prediction:", model.predict(test1_scaled))
print("Test 2 Prediction:", model.predict(test2_scaled))
print("Test 3 Prediction:", model.predict(test3_scaled))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))