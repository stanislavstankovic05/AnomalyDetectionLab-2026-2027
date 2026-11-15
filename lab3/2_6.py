import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

X_train, _ = sklearn.datasets.make_blobs(n_samples=[500, 500], n_features=3, centers=[(0, 10, 0), (10, 0, 10)], cluster_std=1)
X_test = np.random.uniform(-10, 20, (1000, 3))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], s=12)
ax.set_title('date originale')
plt.show()


iforest = IForest(contamination=0.02)
iforest.fit(X_train)
scores = iforest.decision_function(X_test)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores, s=12)
plt.colorbar(sc)
ax.set_title('iforest')
plt.show()


model_dif = DIF(contamination=0.02)
model_dif.fit(X_train)
scores_dif = model_dif.decision_function(X_test)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores_dif, s=12)
plt.colorbar(sc)
ax.set_title('dif')
plt.show()


model_loda = LODA(contamination=0.02)
model_loda.fit(X_train)
scores_loda = model_loda.decision_function(X_test)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores_loda, s=12)
plt.colorbar(sc)
ax.set_title('loda')
plt.show()


def dif_train(hidden_neurons):
    model = DIF(contamination=0.02, hidden_neurons=hidden_neurons)
    model.fit(X_train)
    scores = model.decision_function(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores, s=12)
    plt.colorbar(sc)
    ax.set_title(f'dif hidden={hidden_neurons}')
    plt.show()


def loda_train(n_bins):
    model = LODA(contamination=0.02, n_bins=n_bins)
    model.fit(X_train)
    scores = model.decision_function(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores, s=12)
    plt.colorbar(sc)
    ax.set_title(f'loda bins={n_bins}')
    plt.show()


for index in range(3):
    hidden = [int(np.random.choice([256, 500, 1000])), int(np.random.choice([50, 100, 200]))]
    dif_train(hidden)

for index in range(3):
    bins = int(np.random.randint(3, 101))
    loda_train(bins)
