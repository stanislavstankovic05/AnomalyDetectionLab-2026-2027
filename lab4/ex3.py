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
from pyod.utils.utility import standardizer

data = loadmat('./shuttle.mat')
X, y = data['X'], data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
contamination_rate = count([val for val in y if val==1]) / len(y)
print(contamination_rate)

X_train, X_test = standardizer(X_train, X_test)


#2
#
model_ocsvm = OCSVM(contamination=contamination_rate)
model_ocsvm.fit(X_train)

pred_train = model_ocsvm.labels_
pred_test = model_ocsvm.predict(X_test)
scores_test = model_ocsvm.decision_function(X_test)

ba_test = balanced_accuracy_score(y_test, pred_test)
auc_test = roc_auc_score(y_test, scores_test)
print(f'for ocsvm: ba = {ba_test}\nauc = {auc_test}')

model_deepsvdd = DeepSVDD(n_features=X_train.shape[1], contamination=contamination_rate, verbose=0)
model_deepsvdd.fit(X_train)

pred_train = model_deepsvdd.labels_
pred_test = model_deepsvdd.predict(X_test)
scores_test = model_deepsvdd.decision_function(X_test)

ba_test = balanced_accuracy_score(y_test, pred_test)
auc_test = roc_auc_score(y_test, scores_test)
print(f'for deepsvdd: ba = {ba_test}\nauc = {auc_test}')

# 3

networks = [[32, 16], [64, 32], [128, 64], [256, 128, 64]]
for network in networks:
    model_deepsvdd = DeepSVDD(n_features=X_train.shape[1], hidden_neurons=network, contamination=contamination_rate, verbose=0)
    model_deepsvdd.fit(X_train)

    pred_train = model_deepsvdd.labels_
    pred_test = model_deepsvdd.predict(X_test)
    scores_test = model_deepsvdd.decision_function(X_test)

    ba_test = balanced_accuracy_score(y_test, pred_test)
    auc_test = roc_auc_score(y_test, scores_test)
    print(f'for deepsvdd {network}:\n  - ba = {ba_test}\n  - auc = {auc_test}')
