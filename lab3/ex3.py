import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

X_train, X_test = standardizer(X_train, X_test)


contamination_rate = y.sum() / len(y)

ba_iforest = []
auc_iforest = []
ba_loda = []
auc_loda = []
ba_dif = []
auc_dif = []

for turn in range(10):
    print(f"turn {turn}:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_train, X_test = standardizer(X_train, X_test)

    model_iforest = IForest(contamination=contamination_rate)
    model_iforest.fit(X_train)
    ba_iforest.append(balanced_accuracy_score(y_test, model_iforest.predict(X_test)))
    auc_iforest.append(roc_auc_score(y_test, model_iforest.decision_function(X_test)))

    model_loda = LODA(contamination=contamination_rate)
    model_loda.fit(X_train)
    ba_loda.append(balanced_accuracy_score(y_test, model_loda.predict(X_test)))
    auc_loda.append(roc_auc_score(y_test, model_loda.decision_function(X_test)))

    model_dif = DIF(contamination=contamination_rate)
    model_dif.fit(X_train)
    ba_dif.append(balanced_accuracy_score(y_test, model_dif.predict(X_test)))
    auc_dif.append(roc_auc_score(y_test, model_dif.decision_function(X_test)))

    print("finished")

print(f'iforest: BA = {np.mean(ba_iforest):.4f}, AUC = {np.mean(auc_iforest):.4f}')
print(f'loda: BA = {np.mean(ba_loda):.4f}, AUC = {np.mean(auc_loda):.4f}')
print(f'dif: BA = {np.mean(ba_dif):.4f}, AUC = {np.mean(auc_dif):.4f}')
