import numpy as np
from numpy.ma.core import count
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#1

data = loadmat('./cardio.mat')
X, y = data['X'], data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4)
contamination_rate = count([val for val in y if val==1]) / len(y)
print(contamination_rate)

#2

ocsvm = OneClassSVM()

contamination_rate = count([val for val in y_train if val==1]) / len(y_train)

print(f"contamination arate la : {contamination_rate}")

parameters = [
    {
        'ocsvm__kernel': ['linear'],
        'ocsvm__nu': [0.01, 0.05, contamination_rate, 0.2, 0.5]
    },
    {
        'ocsvm__kernel': ['rbf', 'poly', 'sigmoid'],
        'ocsvm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'ocsvm__nu': [0.01, 0.05, contamination_rate, 0.2, 0.5]
    }
]

#3
# grid = GridSearchCV(ocsvm, parameters, scoring='balanced_accuracy')
# grid.fit(X_train, y_train)
#
# ba_result = balanced_accuracy_score(y_test, grid.best_estimator_.predict(X_test))
# print(f"ba result: {ba_result}")

#4
pipeline = Pipeline([('scaler', StandardScaler()), ('ocsvm', OneClassSVM())])

grid = GridSearchCV(pipeline, parameters, scoring='balanced_accuracy')
grid.fit(X_train, y_train)

ba_result = balanced_accuracy_score(y_test, grid.best_estimator_.predict(X_test))
print(f"ba result: {ba_result}")


#5
label_pyod_train = y_train
label_pyod_test = y_test
label_sklearn_train = (-1)*2 * label_pyod_train + 1
label_sklearn_test = (-1)*2 * label_pyod_test + 1

grid = GridSearchCV(pipeline, parameters, scoring='balanced_accuracy')
grid.fit(X_train, label_sklearn_train)

ba_result = balanced_accuracy_score(label_sklearn_test, grid.best_estimator_.predict(X_test))
print(f"ba result sklearn: {ba_result}")
