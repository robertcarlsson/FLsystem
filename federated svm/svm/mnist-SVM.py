import idx2numpy
import numpy as np

from sklearn import datasets
from sklearn import linear_model

from pathlib import Path

from svm import Federated_SVM

#data_folder = Path('.\dataset\MNIST\\')
#print(data_folder)

#f = open('svm.py')
folder = '.\\federated svm\svm\dataset\MNIST\\'
X = idx2numpy.convert_from_file(folder + 'train-images.idx3-ubyte')
y = idx2numpy.convert_from_file(folder + 'train-labels.idx1-ubyte')
X_test = idx2numpy.convert_from_file(folder + 't10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file(folder + 't10k-labels.idx1-ubyte')


X = np.array([elm.ravel() for elm in X])
X_test = np.array([elm.ravel() for elm in X_test])
print(X.shape)
print(X_test.shape)
number_classifier = linear_model.SGDClassifier(alpha=0.001, max_iter=100, tol=0.0001).fit(X[:2000],y[:2000])

print('accuracy: ', number_classifier.score(X_test, y_test))
print('iterations: ', number_classifier.n_iter_)
#print('SGD: ', number_classifier.avg_sgd_)
federation = Federated_SVM(X, y, X_test, y_test)
