import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Decision Tree classifier using Gini Index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model using Gini Index
clf_gini.fit(X_train, y_train)

# Calculate accuracy for Gini Index
accuracy_gini = clf_gini.score(X_test, y_test)
print(f"Accuracy using Gini Index: {accuracy_gini * 100:.2f}%")

# Create Decision Tree classifier using Information Gain (Entropy)
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)

# Train the model using Information Gain
clf_entropy.fit(X_train, y_train)

# Calculate accuracy for Information Gain
accuracy_entropy = clf_entropy.score(X_test, y_test)
print(f"Accuracy using Information Gain: {accuracy_entropy * 100:.2f}%")

# Feature importance
feature_importances_gini = clf_gini.feature_importances_
feature_importances_entropy = clf_entropy.feature_importances_

print("Feature importances using Gini Index:")
for feature, importance in zip(iris.feature_names, feature_importances_gini):
    print(f"{feature}: {importance:.4f}")

print("\nFeature importances using Information Gain:")
for feature, importance in zip(iris.feature_names, feature_importances_entropy):
    print(f"{feature}: {importance:.4f}")

# Visualize the Decision Tree (Gini Index)
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_gini, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree using Gini Index")
plt.show(block=True)

# Visualize the Decision Tree (Information Gain)
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_entropy, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree using Information Gain")
plt.show(block=True)

# Pruning example by setting a minimal impurity decrease
clf_pruned = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=0.01, random_state=42)
clf_pruned.fit(X_train, y_train)

# Calculate accuracy for pruned tree
accuracy_pruned = clf_pruned.score(X_test, y_test)
print(f"Accuracy using pruned tree (Gini Index with min impurity decrease=0.01): {accuracy_pruned * 100:.2f}%")

# Visualize the pruned Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_pruned, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Pruned Decision Tree using Gini Index")
plt.show(block=True)