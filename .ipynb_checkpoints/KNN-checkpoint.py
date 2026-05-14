import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load dataset
data = pd.read_csv("Iris.csv")

# Remove Id column if present
if 'Id' in data.columns:
    data = data.drop(columns=['Id'])

# Features and target
X = data.drop(columns=['Species'])
y = data['Species']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

best_k = 1
best_acc = 0

# Try different K values
for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)

    # 5-fold cross validation
    scores = cross_val_score(model, X, y, cv=5)

    acc = scores.mean()

    if acc > best_acc:
        best_acc = acc
        best_k = k

print("Best K:", best_k)
print("Cross Validation Accuracy:", best_acc)