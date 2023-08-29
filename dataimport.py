from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Explore the data
print("First few rows of X:")
print(X[:5])  # Print the first 5 rows

print("Basic statistics of X:")
print("Mean:", X.mean(axis=0))
print("Standard Deviation:", X.std(axis=0))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classifying the dataset using RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Classification Report:\n", report)

# Setting the parameter grids
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

classifier = RandomForestClassifier()

grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params)
print(best_model)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

report_best = classification_report(y_test, y_pred_best, target_names=iris.target_names)
print("Best Model Classification Report:\n", report_best)

#Visualisation

# Create a scatter plot
plt.figure(figsize=(8, 6))

for target, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == target, 0], X[y == target, 1], label=target_name)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatter Plot of Iris Dataset')
plt.legend()

plt.show()
