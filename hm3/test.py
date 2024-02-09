import matplotlib.pyplot as plt
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Load the dataset and split it
dt = datasets.load_digits()
X, y = dt.data, dt.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=7)

# Train and evaluate the Logistic Regression model
logistic_regression = LogisticRegression(max_iter=10000)
logistic_regression.fit(X_train, y_train)

y_pred_lr = logistic_regression.predict(X_test)
accuracy_lr = round(metrics.accuracy_score(y_test, y_pred_lr), 2)
print("Accuracy on testing set (Logistic Regression):", accuracy_lr)

# Train and evaluate the MLPClassifier with the initial hidden_layer_sizes=(10,4) configuration
mlp_classifier = MLPClassifier(hidden_layer_sizes=(10, 4), random_state=7, activation='relu', solver='sgd')
mlp_classifier.fit(X_train, y_train)

y_pred_mlp = mlp_classifier.predict(X_test)
accuracy_mlp = round(metrics.accuracy_score(y_test, y_pred_mlp), 4)
print("Accuracy on testing set (MLP with (10,4) hidden layers):", accuracy_mlp)

# Plot the loss curve for the MLPClassifier
plt.plot(mlp_classifier.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Experiment with different hidden_layer_sizes configurations
hidden_layer_sizes_list = [(10,), (10, 4), (20, 10), (30, 20), (50, 30)]

best_accuracy = 0
best_hidden_layers = None

for hidden_layers in hidden_layer_sizes_list:
    mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=7, activation='relu', solver='sgd')
    mlp_classifier.fit(X_train, y_train)
    
    y_pred_mlp = mlp_classifier.predict(X_test)
    accuracy_mlp = metrics.accuracy_score(y_test, y_pred_mlp)
    
    if accuracy_mlp > best_accuracy:
        best_accuracy = accuracy_mlp
        best_hidden_layers = hidden_layers

print("Best hidden_layer_sizes:", best_hidden_layers)
print("Best accuracy on testing set:", round(best_accuracy, 4))





