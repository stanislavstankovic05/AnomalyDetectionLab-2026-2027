import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

contaminatio_rate = 0.07

X, _ = make_blobs(n_samples=[200, 100], centers=[(-10, -10), (10, 10)], cluster_std=[2, 6], random_state=0)

k = int(input())
knn = KNN(n_neighbors=k, contamination=contaminatio_rate).fit(X)
lof = LOF(n_neighbors=k, contamination=contaminatio_rate).fit(X)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[knn.labels_ == 0, 0], X[knn.labels_ == 0, 1], c='blue', s=10)
plt.scatter(X[knn.labels_ == 1, 0], X[knn.labels_ == 1, 1], c='red', s=10)
plt.title(f'knn')

plt.subplot(1, 2, 2)
plt.scatter(X[lof.labels_ == 0, 0], X[lof.labels_ == 0, 1], c='blue', s=10)
plt.scatter(X[lof.labels_ == 1, 0], X[lof.labels_ == 1, 1], c='red', s=10)
plt.title(f'lof')

plt.show()
