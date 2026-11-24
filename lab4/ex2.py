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
        'kernel': ['linear'],
        'nu': [0.01, 0.05, contamination_rate, 0.2, 0.5]
    },
    {
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'nu': [0.01, 0.05, contamination_rate, 0.2, 0.5]
    }
]

#3
