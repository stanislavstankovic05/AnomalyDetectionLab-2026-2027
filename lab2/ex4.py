import numpy as np
from numpy.ma.core import count
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

import copy

data = loadmat('./cardio.mat')
X, y = data['X'], data['y'].ravel()
contamination_rate = count([val for val in y if val==1]) / len(y)
print(contamination_rate)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train_norm = (X_train-mean)/std
    X_test_norm = (X_test-mean)/std

    X_copy = copy.deepcopy(X_train)
    median = np.sort(X_copy, axis=0)[len(X_copy)//2]
    Q3 = np.sort(X_copy, axis=0)[(3*len(X_copy))//4]
    Q1 = np.sort(X_copy, axis=0)[len(X_copy)//4]
    IQR = Q3-Q1
    IQR[IQR == 0] = 1 # divide by 0 error

    X_train_robust = (X_train - median)/IQR
    X_test_robust = (X_test - median)/IQR

    return X_train_norm, X_test_norm, X_train_robust, X_test_robust

X_train_norm, X_test_norm, X_train_robust, X_test_robust = normalize(X_train, X_test)


def train(X_train, X_test):
    knn_train_scores = []
    knn_test_scores = []
    lof_train_scores = []
    lof_test_scores = []

    for k in [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        knn = KNN(n_neighbors=k, contamination=contamination_rate)
        knn.fit(X_train)
        lof = LOF(n_neighbors=k, contamination=contamination_rate)
        lof.fit(X_train)

        knn_train_scores.append(knn.decision_scores_)
        knn_test_scores.append(knn.decision_function(X_test))
        lof_train_scores.append(lof.decision_scores_)
        lof_test_scores.append(lof.decision_function(X_test))

        print(f'for k={k} knn ba train={balanced_accuracy_score(y_train, knn.labels_)} \n  test={balanced_accuracy_score(y_test, knn.predict(X_test))}')
        print(f'for k={k} lof bal train={balanced_accuracy_score(y_train, lof.labels_)} \n test={balanced_accuracy_score(y_test, lof.predict(X_test))}')

    return knn_train_scores, knn_test_scores, lof_train_scores, lof_test_scores


knn_train_scores, knn_test_scores, lof_train_scores, lof_test_scores = train(X_train_norm, X_test_norm)
knn_standard = standardizer(np.array(knn_train_scores).T, np.array(knn_test_scores).T)
knn_train_standard = knn_standard[0]
knn_test_standard = knn_standard[1]

lof_standard = standardizer(np.array(lof_train_scores).T, np.array(lof_test_scores).T)
lof_train_standard = lof_standard[0]
lof_test_standard = lof_standard[1]

knn_average_train = average(knn_train_standard)
knn_average_test = average(knn_test_standard)
knn_average_threshold_train = np.quantile(knn_average_train, 1 - contamination_rate)
knn_average_threshold_test = np.quantile(knn_average_test, 1 - contamination_rate)

knn_average_pred_train = knn_average_train > knn_average_threshold_train
knn_average_pred_test = knn_average_test > knn_average_threshold_test

print(balanced_accuracy_score(y_train, knn_average_pred_train))
print(balanced_accuracy_score(y_test, knn_average_pred_test))
