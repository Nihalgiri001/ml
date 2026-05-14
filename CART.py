import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Iris.csv")

# Features and target
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# Encode labels (Species is categorical)
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# CART (Gini) Decision Tree
model = DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)
# Prediction on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"CART (Gini) Accuracy: {accuracy * 100:.2f}%")

# Plotting the decision tree with better labeling
plt.figure(figsize=(15,10))
plot_tree(model,
filled=True,
feature_names=X.columns,
class_names=le.classes_,
rounded=True,
fontsize=10,
precision=2,
label='all') # Show all details in the nodes
plt.title("CART (Gini) Decision Tree")
plt.show()