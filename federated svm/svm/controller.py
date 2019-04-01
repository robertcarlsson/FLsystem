import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model

from generator import make_class_dataset, make_class_dataset_test
from svm import SVM

n_svm = 4
sample_base = 500
samples = n_svm * sample_base

X, y, X_test, y_test = make_class_dataset_test(
    samples=samples,
    test_ratio=0.2,
    n_features=2,
    random_seed=4
)

X_list = np.split(X, n_svm)
y_list = np.split(y, n_svm)

svm_list = []

for i in range(n_svm):
    svm_list.append(SVM(X_list[i], y_list[i]))



for svm in svm_list:
    svm.partial_fit()
    print(svm.get_score(X_test, y_test))

coef = svm_list[0].clf.coef_
intercept = svm_list[0].clf.intercept_

#print(coef)
#print(intercept)

for svm in svm_list[1:]:
    coef += svm.clf.coef_
    intercept += svm.clf.intercept_
coef = coef / n_svm
intercept = intercept / n_svm

#print(coef)
#print(intercept)

global_svm_wh = SVM(X, y)
global_svm_wh.partial_fit(coef=coef, intercept=intercept)
print(global_svm_wh.get_score(X_test, y_test))

global_svm = SVM(X, y)
global_svm.partial_fit()
print(global_svm.get_score(X_test, y_test))

