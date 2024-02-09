import matplotlib.pyplot as plt
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split

# Load the Wine dataset
wine = datasets.load_wine()
X, y = wine.data, wine.target

# Define the classifier (Decision Tree with entropy)
predictor = tree.DecisionTreeClassifier(criterion='entropy')

# Split the data
train_size = 0.7

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=7)

# Train the classifier on the training data
predictor.fit(X_train, y_train)

# Plot the decision tree
figure = plt.figure(figsize=(12, 6))
tree.plot_tree(predictor, feature_names=wine.feature_names, class_names=wine.target_names, impurity=False)
plt.show()
