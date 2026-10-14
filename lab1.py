import pyod.utils.data as od
from pyod.models.knn import KNN as knn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, roc_auc_score
import sklearn.metrics as skm
import numpy as np
#1
x, x_t, y, y_t = od.generate_data(n_train= 400,n_test= 100, contamination=0.5)
print(x.shape, x_t.shape, y.shape, y_t.shape)

inliers = x[y == 0]
out = x[y == 1]
# plt.scatter(inliers[:, 0], inliers[:, 1], c="tab:blue")
# plt.scatter(out[:, 0], out[:, 1], c="tab:red")
# plt.scatter(x[:,0], x[:,1], c=y)
# plt.show()

#2

model = knn(contamination=0.5)

model.fit(x)
pred_train = model.predict(x)
pred_test = model.predict(x_t)


mat1 = skm.confusion_matrix(y, pred_train)
mat2 = skm.confusion_matrix(y_t, pred_test)

print(f"{5*'='}")
print(mat1)
print()
print(mat2)

print(f"{5*'='}")
(tn, fp), (fn, tp) = mat1
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
balenced_score1 = (tpr + tnr) / 2
print(balenced_score1)
score1 = model.decision_function(x)

curve1 = skm.roc_curve(y, score1)

(tn, fp), (fn, tp) = mat2
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
balenced_score2 = (tpr + tnr) / 2
print(balenced_score2)
score2 = model.decision_function(x_t)
curve2 = skm.roc_curve(y_t, score2)

fpr, tpr, _ = skm.roc_curve(y_t, score2)
plt.plot(fpr, tpr)
plt.show()

#3
x_train, x_test, y_train, y_test = od.generate_data(n_train= 1000,n_test= 0, contamination=0.1, n_features=1)

z_score = (x_train - np.mean(x_train))/np.std(x_train)

print(f"zscore = {z_score}")
threshold = np.quantile(z_score, 0.1)

anomalies = (x_train < threshold)

print(f"anomalies = ", anomalies)


(tn, fp), (fn, tp) = skm.confusion_matrix(y_train, anomalies)
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
balenced_score3 = (tpr + tnr) / 2
print(balenced_score3)

print(threshold, balenced_score3)

#4
mean = np.array([6, 9, 7])
sigma = np.array([[7, 2, 3],
                  [2, 6, 2],
                  [3, 2, 5]])

x = np.random.standard_normal((3, 1000))
L = np.linalg.cholesky(sigma)
y = L @ x + mean.reshape(3, 1)

z_score = np.sqrt(np.sum((y - y.mean(axis=1, keepdims=True)) * (np.linalg.inv(np.cov(y)) @ (y - y.mean(axis=1, keepdims=True))), axis=0))

print(f"zscore: {z_score}")
