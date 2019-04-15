import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model

from generator import make_class_dataset, make_class_dataset_test
from svm import SVM

n_svm = 4
sample_base = 500
samples = n_svm * sample_base

all_scores = []


def svm_iterations(i):
    
    X, y, X_test, y_test = make_class_dataset_test(
        samples=samples,
        test_ratio=0.2,
        n_features=2,
        random_seed=i*2
    )

    X_list = np.split(X, n_svm)
    y_list = np.split(y, n_svm)

    svm_list = []

    for i in range(n_svm):
        svm_list.append(SVM(X_list[i], y_list[i]))


    scores = []
    for svm in svm_list:
        svm.partial_fit()
        scores.append(svm.get_score(X_test, y_test))

    high_index = np.where(scores == np.amax(scores))[0][0]


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

    global_svm_avg = SVM(X, y)
    global_svm_high = SVM(X, y)
    global_svm = SVM(X, y)


    global_svm_avg.partial_fit(coef=coef, intercept=intercept)
    global_svm_high.partial_fit(coef=svm_list[high_index].clf.coef_, intercept=svm_list[high_index].clf.intercept_)
    global_svm.partial_fit()
    #print("Index: ", high_index)
    #for svm in scores:
    #    print('Sub-SVM: ', svm)

    scores.append(global_svm_avg.get_score(X_test, y_test))
    scores.append(global_svm_high.get_score(X_test, y_test))
    scores.append(global_svm.get_score(X_test, y_test))
    #print('Avarage: ', global_svm_avg.get_score(X_test, y_test))
    #print('Highest: ', global_svm_high.get_score(X_test, y_test))
    #print('Control: ', global_svm.get_score(X_test, y_test))
    all_scores.append(scores)

iters = 100
for i in range(iters):
    svm_iterations(i)
#print(all_scores)

x = np.arange(0, iters, 1)
all_scores = np.array(all_scores)
print(all_scores.shape)
#plt.plot(x, all_scores[:,0:1])
#plt.plot(x, all_scores[:,1:2])
#plt.plot(x, all_scores[:,2:3])
#plt.plot(x, all_scores[:,3:4])
plt.plot(x, all_scores[:,4:5], 'ro')
plt.plot(x, all_scores[:,5:6], 'bs')
plt.plot(x, all_scores[:,6:7], 'g^')

avg = []
avg.append(np.mean(all_scores[:,4:5]))
avg.append(np.mean(all_scores[:,5:6]))
avg.append(np.mean(all_scores[:,6:7]))


for a in avg:
    print(a)
plt.ylabel('Accuracy')
plt.xlabel('Random seed for iteration')
plt.show()