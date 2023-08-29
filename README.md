# Description

Build a classifier to predict the species of iris flowers based on their petal and sepal measurements.

## Libraries Used

- sci-kit learn
- pandas
- numpy

## Toolkit Used

- Jupyter Notebook

---

### Steps

1. Understand the Problem Statement
2. Gather the Dataset - https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
3. Preprocess the Data
4. Explore the Data
5. Choose a classifier
6. Training the classifier
7. Evaluation the model
8. Tuning Hyperparameters
9. Visualization

## Problem Statement

The aim of this project is to predict the species of iris flowers based on their petal and sepal measurements. The Iris dataset contains four features (sepal length, sepal width, petal length, petal width) and one target variable (species label). Each sample is associated with one of three species: setosa, versicolor, or virginica. (Shown in figure)

![*Iris setosa*](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ef2ad752-12ff-4007-afce-07eadc9854c5/Untitled.png)

*Iris setosa*

![*Iris versicolor*](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/175f9ea5-3f46-4203-be82-b6485cb48c3e/Untitled.png)

*Iris versicolor*

![*Iris Virginca*](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/91b66d47-4cec-4932-9bd3-6db8209d5892/Untitled.png)

*Iris Virginca*

Each sample is represented by its features (numerical measurements) and associated with a target label.

---

## Dataset

The dataset is imported from the scikit-learn library. The following python code is used to import the data into our Jupyter Notebook.

```python

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # Feature matrix (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target vector (species labels)

```

More information about this dataset is given in https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

## Preprocessing and Exploring the data

Preprocessing the data is a critical step in machine learning projects. It involves cleaning, transforming, and preparing the data to be suitable for training a model. In the context of the iris flower classification project, here's how we can preprocess the data:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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
```

The following modules are imported:

- **`load_iris`**: To load the Iris dataset.
- **`train_test_split`**: To split the data into training and testing sets.
- **`StandardScaler`**: To scale the features.
- **`RandomForestClassifier`**: The classification algorithm used.
- **`classification_report`**: To generate a classification report for evaluating the model.

### Training and Testing Splits

The dataset is split into training and testing sets using the **`train_test_split()`** function.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

This is done to ensure we have data to both train and evaluate your model's performance.

### Scaling Feature

We use the **`StandardScaler`** to scale the features. The training data is fit to the scaler and transformed, while the testing data is only transformed based on the training data's statistics.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaling ensures that the features have similar ranges, which can improve the performance of some machine learning algorithms.

### Model Training and Evaluation

We create an instance of **`RandomForestClassifier`** and fit it to the scaled training data. Then, we make predictions on the scaled testing data and generate a classification report using **`classification_report()`**.

```python
classifier = RandomForestClassifier()
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Classification Report:\n", report)
```

The classification report provides metrics like precision, recall, F1-score, and support for each class, giving us a comprehensive view of your model's performance.

```python
First few rows of X:
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
Basic statistics of X:
Mean: [5.84333333 3.05733333 3.758      1.19933333]
Standard Deviation: [0.82530129 0.43441097 1.75940407 0.75969263]
Classification Report:
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

The output of the program is shown.

## Hyperparameter Tuning

Hyperparameter tuning involves finding the best combination of hyperparameters for your machine learning model to achieve optimal performance.

### Identifying Hyperparameters

We Identify the hyperparameters of the model we want to tune. These are the parameters that we set before training and affect the learning process. For example, in a Random Forest classifier, hyperparameters include the number of trees, maximum depth of trees, and minimum samples per leaf.

### Defining Hyperparameter Search Space

We determine the range or values we want to search for each hyperparameter. This defines the search space for the tuning process. We can define it manually or use tools like **`GridSearchCV`** or **`RandomizedSearchCV`** from scikit-learn.

### Tuning Strategy

There are two main strategies for hyperparameter tuning:

- **Grid Search:** In this method, you provide a set of possible values for each hyperparameter, and the algorithm exhaustively tries all possible combinations. It's suitable for a small search space.
- **Random Search:** This method randomly samples hyperparameters from the defined search space. It's often more efficient than grid search for larger search spaces.

```python
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
```

The above code is run to determine the best parameters and the best model.

### Evaluation of best model

Now that we have the best model, we evaluate its performance on the test set to see how well it generalizes.We use the **`X_test_scaled`** and **`y_test`** data that we previously split.

```python
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

report_best = classification_report(y_test, y_pred_best, target_names=iris.target_names)
print("Best Model Classification Report:\n", report_best)
```

## Visualisation

This code will create a scatter plot of the Iris dataset, where different species are represented by different colors

```python
# Create a scatter plot
plt.figure(figsize=(8, 6))

for target, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == target, 0], X[y == target, 1], label=target_name)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatter Plot of Iris Dataset')
plt.legend()

plt.show()
```

The x-axis represents sepal length and the y-axis represents sepal width. Each data point is colored based on its corresponding species label (setosa, versicolor, virginica).

![Scatter Plot of Iris Dataset](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d4eeee82-f378-4b73-91bb-4d50aa5c756c/Figure_1.png)

Scatter Plot of Iris Dataset

# Conclusion

In this project, we aimed to classify Iris flower species based on their petal and sepal measurements. We followed a systematic pipeline, including data loading, preprocessing, model training, hyperparameter tuning, and visualization. Here's a summary of the key steps and findings:

1. **Data Exploration:**
We loaded the Iris dataset, examined the first few rows of the data, and calculated basic statistics like mean and standard deviation for each feature.
2. **Data Preprocessing:**
We split the data into training and testing sets, scaled the features using StandardScaler to ensure similar ranges, and prepared the dataset for model training.
3. **Model Training and Evaluation:**
We trained a Random Forest classifier on the scaled training data and evaluated its performance using a classification report. This report provided precision, recall, F1-score, and support for each class.
4. **Hyperparameter Tuning:**
Utilizing GridSearchCV, we explored different hyperparameter combinations for the Random Forest classifier. This led us to discover the best hyperparameters and the best model configuration.
5. **Best Model Evaluation:**
We evaluated the performance of the best-tuned model on the test set. The updated classification report showed how hyperparameter tuning affected model performance.
6. **Data Visualization:**
We created a scatter plot to visualize the distribution of Iris flower species based on sepal length and width. This visualization helps us understand the separation of different species.

**Key Takeaways:**

- Hyperparameter tuning significantly impacts model performance and can lead to better results.
- Visualization helps us gain insights into the data distribution and the model's predictions.

This project demonstrates the entire machine learning workflow, from data preprocessing to model evaluation and visualization. It's important to note that this is just one approach, and further exploration and experimentation could yield even better results or insights.

- Entire Code
    
    ```python
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
    ```
    

# Author

[Arjith Praison](https://www.linkedin.com/in/arjith-praison-95b145184/)

University of Siegen
Germany
