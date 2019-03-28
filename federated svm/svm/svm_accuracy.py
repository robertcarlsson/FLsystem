
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.datasets import load_svmlight_files

X_train, y_train, X_test, y_test = load_svmlight_files(
    ("./dataset/a9a.txt", "./dataset/a9a.t"))

print(type(X_train))
print(y_train.shape)
print(X_test.shape)
#print(X_test[0], y_test[0])
X_train = np.array(X_train)
# Shuffle
idx = np.arange(X_train.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]

# Standardize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std

# Step size in the mesh
h = .02 

clf = linear_model.SGDClassifier(alpha=0.001, max_iter=100, tol=0.0001).fit(X_train,y_train)

# create a mesh to plot in
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

