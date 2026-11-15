import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

X_train_3d, _ = sklearn.datasets.make_blobs(n_samples=[500, 500], n_features=3, centers=[(0, 10, 0), (10, 0, 10)], cluster_std=1)
X_test_3d = np.random.uniform(-10, 20, (1000, 3))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2], s=12)
ax.set_title('date originale 3d')
plt.show()


iforest_3d = IForest(contamination=0.02)
iforest_3d.fit(X_train_3d)
scores_iforest_3d = iforest_3d.decision_function(X_test_3d)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=scores_iforest_3d, s=12)
plt.colorbar(sc)
ax.set_title('iforest 3d')
plt.show()


model_dif_3d = DIF(contamination=0.02)
model_dif_3d.fit(X_train_3d)
scores_dif_3d = model_dif_3d.decision_function(X_test_3d)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=scores_dif_3d, s=12)
plt.colorbar(sc)
ax.set_title('dif 3d')
plt.show()


model_loda_3d = LODA(contamination=0.02)
model_loda_3d.fit(X_train_3d)
scores_loda_3d = model_loda_3d.decision_function(X_test_3d)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=scores_loda_3d, s=12)
plt.colorbar(sc)
ax.set_title('loda 3d')
plt.show()


def dif_train_3d(hidden_neurons):
    model = DIF(contamination=0.02, hidden_neurons=hidden_neurons)
    model.fit(X_train_3d)
    scores = model.decision_function(X_test_3d)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=scores, s=12)
    plt.colorbar(sc)
    ax.set_title(f'dif 3d hidden={hidden_neurons}')
    plt.show()


def loda_train_3d(n_bins):
    model = LODA(contamination=0.02, n_bins=n_bins)
    model.fit(X_train_3d)
    scores = model.decision_function(X_test_3d)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=scores, s=12)
    plt.colorbar(sc)
    ax.set_title(f'loda 3d bins={n_bins}')
    plt.show()


for index in range(3):
    n_layers = np.random.randint(2, 4)
    hidden = [int(np.random.choice([4, 8, 16, 32, 64, 128])) for _ in range(n_layers)]
    dif_train_3d(hidden)

for index in range(3):
    bins = int(np.random.randint(3, 101))
    loda_train_3d(bins)
