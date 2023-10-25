import numpy as np # In order to calculate the mean and the standard deviation.
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

"""
Author: embid13
Version: v0.1
Date: 10/22/2023
Description: This program is divided into two parts. First, a "CrossValidator" class is defined to perform cross-validation. Then, a function is defined that conducts a grid search with a predefined hyperparameter space. In both parts, custom functions are compared with the functions from the sklearn library.
To execute this file, use Python. I'm using Python 3.11.4.
	- python3 crossValidatorExercises.py
"""

# CrossValidator class
class CrossValidator:
    """
    1.a.i)
    Constructor.
    Initialize the class with the provided parameters.

    Args:
    dataset (object): The dataset used for the machine learning task.
    n_folds (int): The number of folds for cross-validation.
    algorithm (str): The machine learning algorithm to be applied.
    """
    def __init__(self, dataset, n_folds, algorithm):
        self.dataset = dataset
        self.n_folds = n_folds
        self.algorithm = algorithm

    """
    for each fold it creates training and test sets.

    Returns:
    - fold_sets: a list of tuples (training/testing) for each fold.
    """
    # 1.a.ii)
    def get_folds(self):
        fold_sets = []
        # 1.a.iv) "shuffle the dataset"
        data, labels = shuffle(self.dataset.data, self.dataset.target, random_state=42)

        fold_size = len(data) // self.n_folds # folds' size

        for i in range(self.n_folds):
            comienzo = i * fold_size
            fin = (i + 1) * fold_size
            test_indices = list(range(comienzo, fin))
            train_indices = list(range(0, comienzo)) + list(range(fin, len(data)))
            X_train, X_test = data[train_indices], data[test_indices]
            y_train, y_test = labels[train_indices], labels[test_indices]

            fold_sets.append((X_train, X_test, y_train, y_test))

        return fold_sets

    """
    It carries out the cross validation.

    Returns:
    - mean_score: the mean of the calculated scores.
    - std_score: standard deviation of the scores.
    """
    # 1.a.iii)
    def cross_validate(self):
        scores = []

        for X_train, X_test, y_train, y_test in self.get_folds():
            # 1.a.iii.1)
            model = self.algorithm.fit(X_train, y_train)
            y_pred = model.predict(X_test)  # predict the target.
            # 1.a.iii.2)
            score = accuracy_score(y_test, y_pred)  # calculate the precision.
            scores.append(score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        return mean_score, std_score


iris = load_iris() # Load the dataset.
# Create a new instance with the "RandomForestClassifier" algorithm.
algorithm = RandomForestClassifier(n_estimators=100, random_state=42)
n_folds = 5
# 1.b)
cv = CrossValidator(dataset=iris, n_folds=n_folds, algorithm=algorithm) # Create an instance of the class "CrossValidator".

# Carry out the cross validation and get the mean and the standard deviation of the scores.
mean_score, std_score = cv.cross_validate()

print("Scores with the custom cross validation:")
print(f"Average of the scores with custom cross validation: {mean_score}")
print(f"Standard deviation with custom cross validation: {std_score}")


# 1.c)
scores = cross_val_score(algorithm, iris.data, iris.target, cv=n_folds, scoring='accuracy') 
# Carry out the cross validation with sk-learn's function.

# Calculate the average and the standard deviation of the scores.
mean_score = scores.mean()
std_score = scores.std()

print("Scores with the sk-learn's cross validation:")
print(f"Average of the scores with sk-learn's cross validation: {mean_score}")
print(f"Standard deviation with sk-learn's cross validation: {std_score}")


# exercise 2, grid search.
def grid_search(param_grid, dataset):
    best_params = {}
    best_score = 0.0

    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    algorithm = RandomForestClassifier( # we are only going to evaluate randomForestClassifier
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42
                    )
                    # Create an instance of the class Cross Validator.
                    cv = CrossValidator(dataset=dataset, n_folds=5, algorithm=algorithm)

                    # Perform cross-validation and obtain the average performance.
		    # The standard deviation could provide an evaluation of the performance variability of the problem,
		    # but we will not assess it in this case. (This could help us find the "ideal parameters" more quickly.)
                    mean_score, _ = cv.cross_validate()

                    # If a better score is achieved, save them.
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf
                        }

    return best_params, best_score

# Define the hyperparameters' pool.
params_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

best_params, best_score = grid_search(params_grid, iris) # custom grid search

print("-#################################################-")
print("Best parameters found with custom grid search:")
print(best_params)
print(f"Best score found with custom grid search: {best_score:.2f}")


# Create an instance of the selected algorithm, in our case "RandomForestClassifier".
algorithm = RandomForestClassifier(random_state=42)

# Create an instance of GridSearchCV (sk-learn)
grid_search = GridSearchCV(algorithm, params_grid, scoring='accuracy', cv=n_folds) # función sklearn
data = iris.data
labels = iris.target
# Fit the model with GridSearchCV (sk-learn)
grid_search.fit(data, labels)


# Obtain the best parameters and the best score:
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters found with the sk-learn's grid search:")
print(best_params)
print(f"Best score found with the sk-learn's grid search: {best_score:.2f}")

