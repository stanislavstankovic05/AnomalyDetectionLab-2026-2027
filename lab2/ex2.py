import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from pyod.models.knn import KNN
from pyod.utils.data import generate_data_clusters

contamination = 0.1
X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=contamination
)

for k in [10 * k for k in range(1, 10)]:
    model = KNN(n_neighbors=k, contamination=contamination)
    model.fit(X_train)

    pred_train = model.labels_
    pred_test = model.predict(X_test)

    ba_train = balanced_accuracy_score(y_train, pred_train)
    ba_test = balanced_accuracy_score(y_test, pred_test)
    print(f'k={k}  BA train={ba_train:.3f}  BA test={ba_test:.3f}')

    plt.figure(figsize=(10, 9))

    plt.subplot(2, 2, 1)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', s=10)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', s=10)
    plt.title('Ground truth train')

    plt.subplot(2, 2, 2)
    plt.scatter(X_train[pred_train == 0, 0], X_train[pred_train == 0, 1], c='blue', s=10)
    plt.scatter(X_train[pred_train == 1, 0], X_train[pred_train == 1, 1], c='red', s=10)
    plt.title(f'Predicted train (k={k}, BA={ba_train:.2f})')

    plt.subplot(2, 2, 3)
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='blue', s=10)
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='red', s=10)
    plt.title('Ground truth test')

    plt.subplot(2, 2, 4)
    plt.scatter(X_test[pred_test == 0, 0], X_test[pred_test == 0, 1], c='blue', s=10)
    plt.scatter(X_test[pred_test == 1, 0], X_test[pred_test == 1, 1], c='red', s=10)
    plt.title(f'Predicted test (k={k}, BA={ba_test:.2f})')

    plt.tight_layout()

plt.show()
