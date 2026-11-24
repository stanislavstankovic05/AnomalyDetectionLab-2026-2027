import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.data import generate_data
from pyod.models.deep_svdd import DeepSVDD

#1
contamination_rate = 0.15
X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=200, n_features=3, contamination=contamination_rate)

model_deepsvdd = DeepSVDD(n_features=3, contamination=contamination_rate, verbose=0)
model_deepsvdd.fit(X_train)
y_train_pred = model_deepsvdd.labels_
y_test_pred = model_deepsvdd.predict(X_test)
scores = model_deepsvdd.decision_function(X_test)

print(f'for deepsvdd: ba = {balanced_accuracy_score(y_test, y_test_pred):.4f}\nauc = {roc_auc_score(y_test, scores):.4f}')

plt.figure()
plt.axes(projection='3d')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], zs=X_train[y_train == 0, 2], s=20)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], zs=X_train[y_train == 1, 2], s=20)
plt.title('train ground truth')

plt.figure()
plt.axes(projection='3d')
plt.scatter(X_train[y_train_pred == 0, 0], X_train[y_train_pred == 0, 1], zs=X_train[y_train_pred == 0, 2], s=20)
plt.scatter(X_train[y_train_pred == 1, 0], X_train[y_train_pred == 1, 1], zs=X_train[y_train_pred == 1, 2], s=20)
plt.title('train predicted')

plt.figure()
plt.axes(projection='3d')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], zs=X_test[y_test == 0, 2], s=20)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], zs=X_test[y_test == 1, 2], s=20)
plt.title('test ground truth')

plt.figure()
plt.axes(projection='3d')
plt.scatter(X_test[y_test_pred == 0, 0], X_test[y_test_pred == 0, 1], zs=X_test[y_test_pred == 0, 2], s=20)
plt.scatter(X_test[y_test_pred == 1, 0], X_test[y_test_pred == 1, 1], zs=X_test[y_test_pred == 1, 2], s=20)
plt.title('test predicted')

plt.show()
