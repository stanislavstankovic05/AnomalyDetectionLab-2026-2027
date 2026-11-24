import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

#1
contamination_rate = 0.15
X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=200, n_features=3, contamination=contamination_rate)

#2
model_linear = OCSVM(kernel='linear', contamination=contamination_rate)
model_linear.fit(X_train)
y_train_pred = model_linear.labels_
y_test_pred = model_linear.predict(X_test)
scores = model_linear.decision_function(X_test)

print(f'ba = {balanced_accuracy_score(y_test, y_test_pred):.4f}\n roc auc = {roc_auc_score(y_test, scores):.4f}')


#3

# train data with ground thruth
plt.figure()
plt.axes(projection='3d')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], zs=X_train[y_train == 0, 2], s=20)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], zs=X_train[y_train == 1, 2], s=20)
plt.title('train data with ground thruth')

# train with predcited result
plt.figure()
plt.axes(projection='3d')
plt.scatter(X_train[y_train_pred == 0, 0], X_train[y_train_pred == 0, 1], zs=X_train[y_train_pred == 0, 2], s=20)
plt.scatter(X_train[y_train_pred == 1, 0], X_train[y_train_pred == 1, 1], zs=X_train[y_train_pred == 1, 2], s=20)
plt.title('train with predicted result')

# test data with predcited result
plt.figure()
plt.axes(projection='3d')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], zs=X_test[y_test == 0, 2], s=20)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], zs=X_test[y_test == 1, 2], s=20)
plt.title('test data with predcited result')

#test data with predicted results
plt.figure()
plt.axes(projection='3d')
plt.scatter(X_test[y_test_pred == 0, 0], X_test[y_test_pred == 0, 1], zs=X_test[y_test_pred == 0, 2], s=20)
plt.scatter(X_test[y_test_pred == 1, 0], X_test[y_test_pred == 1, 1], zs=X_test[y_test_pred == 1, 2], s=20)
plt.title('test data with predcited result')

plt.show()

#4
model_rbf = OCSVM(kernel='rbf', contamination=contamination_rate)
model_rbf.fit(X_train)
y_train_pred = model_rbf.labels_
y_test_pred = model_rbf.predict(X_test)
scores = model_rbf.decision_function(X_test)

print(f'for ocsvm rbf: ba = {balanced_accuracy_score(y_test, y_test_pred):.4f}\nauc = {roc_auc_score(y_test, scores):.4f}')

# train data with ground thruth
plt.figure()
plt.axes(projection='3d')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], zs=X_train[y_train == 0, 2], s=20)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], zs=X_train[y_train == 1, 2], s=20)
plt.title('train data with ground thruth')

# train with predcited result
plt.figure()
plt.axes(projection='3d')
plt.scatter(X_train[y_train_pred == 0, 0], X_train[y_train_pred == 0, 1], zs=X_train[y_train_pred == 0, 2], s=20)
plt.scatter(X_train[y_train_pred == 1, 0], X_train[y_train_pred == 1, 1], zs=X_train[y_train_pred == 1, 2], s=20)
plt.title('train with predicted result')

# test data with predcited result
plt.figure()
plt.axes(projection='3d')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], zs=X_test[y_test == 0, 2], s=20)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], zs=X_test[y_test == 1, 2], s=20)
plt.title('test data with predcited result')

#test data with predicted results
plt.figure()
plt.axes(projection='3d')
plt.scatter(X_test[y_test_pred == 0, 0], X_test[y_test_pred == 0, 1], zs=X_test[y_test_pred == 0, 2], s=20)
plt.scatter(X_test[y_test_pred == 1, 0], X_test[y_test_pred == 1, 1], zs=X_test[y_test_pred == 1, 2], s=20)
plt.title('test data with predcited result')

plt.show()
