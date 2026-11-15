from numpy.ma import anom
import pyod.utils.data as od
from pyod.models.knn import KNN as knn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, roc_auc_score
import sklearn.metrics as skm
import numpy as np
import sklearn


def compute_anomaly_score(train_projected, test_points, bins=10):
    anomaly_score = np.zeros(len(test_points))

    for i in range(train_projected.shape[1]):
        hist, bin_edges = np.histogram(train_projected[:, i], bins=bins, range=(-5, 5))
        probabilities = hist / hist.sum()

        bin_index = np.digitize(test_points[:, i], bin_edges) - 1
        anomaly_score += probabilities[bin_index]

    return anomaly_score / train_projected.shape[1]

X_train, _ = sklearn.datasets.make_blobs(n_samples=500, n_features=2, centers=[(0, 0)], cluster_std=1)

#print(X_train)

projection_vectors =   np.random.multivariate_normal(mean=[0,0], cov=np.eye(2), size=5)
projection_vectors = projection_vectors/np.linalg.norm(projection_vectors, axis=1, keepdims=True)
X_train_projected = X_train @ projection_vectors.T

anomaly_score_train = compute_anomaly_score(X_train_projected, X_train_projected, 10)
print(anomaly_score_train)


X_test = np.random.uniform(-3, 3, (500, 2))
X_test_projected = X_test @ projection_vectors.T
anomaly_score_test = compute_anomaly_score(X_train_projected, X_test_projected, 10)

print(anomaly_score_test)

for k in [5, 10, 15, 25, 50]:
    anomaly_score_test_k = compute_anomaly_score(X_train_projected, X_test_projected, bins=k)
    print(anomaly_score_test_k)

    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=anomaly_score_test_k)
    plt.colorbar()
    plt.title(f'bins={k}')

plt.show()
