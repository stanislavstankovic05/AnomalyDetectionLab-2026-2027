from numpy.ma import anom
import pyod.utils.data as od
from pyod.models.knn import KNN as knn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, roc_auc_score
import sklearn.metrics as skm
import numpy as np
import sklearn
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

def dif_train(hidden_neurons):
    model = DIF(contamination=0.02, hidden_neurons=hidden_neurons)
    model.fit(X_train)
    scores = model.decision_function(X_test)

    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=scores, s=12)
    plt.colorbar()
    plt.title(f'dif hidden={hidden_neurons}')
    plt.show()


def loda_train(n_bins):
    model = LODA(contamination=0.02, n_bins=n_bins)
    model.fit(X_train)
    scores = model.decision_function(X_test)

    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=scores, s=12)
    plt.colorbar()
    plt.title(f'loda bins={n_bins}')
    plt.show()


X_train, _ = sklearn.datasets.make_blobs(n_samples=[500, 500], n_features=2, centers=[(10, 0), (0, 10)], cluster_std=1)

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], s=12)
plt.title('date originale')
plt.show()

iforest = IForest(contamination=0.02)
iforest.fit(X_train)
X_test = np.random.uniform(-10, 20, (1000, 2))

scores = iforest.decision_function(X_test)

plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=scores, s=12)
plt.colorbar()
plt.title('iforest')
plt.show()


model_dif = DIF(contamination=0.02)
model_dif.fit(X_train)
scores_dif = model_dif.decision_function(X_test)

plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=scores_dif, s=12)
plt.colorbar()
plt.title('dif')
plt.show()


model_loda = LODA(contamination=0.02)
model_loda.fit(X_train)
scores_loda = model_loda.decision_function(X_test)

plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=scores_loda, s=12)
plt.colorbar()
plt.title('loda')
plt.show()


# 5

for index in range(3):
    hidden = [int(np.random.choice([256, 500, 1000])), int(np.random.choice([50, 100, 200]))]
    dif_train(hidden)

for index in range(3):
    bins = int(np.random.randint(3, 101))
    loda_train(bins)
