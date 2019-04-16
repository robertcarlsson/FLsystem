import idx2numpy
import numpy as np

from sklearn import linear_model
from sklearn.datasets import load_digits

from svm import Federated_SVM, SVM

def scikit_digits():

    digits = load_digits()
    print('Shapes:', digits.target.shape)

    pivot_point = 1500

    digits_learn_data, digits_learn_target, \
    digits_validate_data, digits_validate_target = \
        digits.data[:pivot_point], digits.target[:pivot_point], \
        digits.data[pivot_point:], digits.target[pivot_point:]

    n_svm = 4

    X_list = np.split(digits_learn_data[:1000], n_svm)
    y_list = np.split(digits_learn_target[:1000], n_svm)

    federation = Federated_SVM(
        digits_validate_data, 
        digits_validate_target, 
        global_aggregation=False)

    for i in range(n_svm):
        federation.add_participant(SVM(X_list[i], y_list[i], 5334, 1))

    federation.run_eon(15)

    print(federation.all_scores)
    print(y_list[1][:100])

def mnist_digits():

    folder = '.\\federated svm\dataset\MNIST\\'
    X = idx2numpy.convert_from_file(folder + 'train-images.idx3-ubyte')
    y = idx2numpy.convert_from_file(folder + 'train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file(folder + 't10k-images.idx3-ubyte')
    y_test = idx2numpy.convert_from_file(folder + 't10k-labels.idx1-ubyte')

    X = np.array([elm.ravel() for elm in X])
    X_test = np.array([elm.ravel() for elm in X_test])

    n_svm = 4
    n_datapoints = 12000
    X_list = np.split(X[:n_datapoints], n_svm)
    y_list = np.split(y[:n_datapoints], n_svm)

    X_list = [X_list[0][:40], X_list[1][:500], X_list[2][:1000], X_list[3][:3000]]
    y_list = [y_list[0][:40], y_list[1][:500], y_list[2][:1000], y_list[3][:3000]]
    
    federation = Federated_SVM(
        X_test, 
        y_test, 
        global_aggregation=True)

    for i in range(n_svm):
        federation.add_participant(SVM(X_list[i], y_list[i], 5333, 3))

    federation.run_eon(12)

    print(y_list[1][:100])
    print(federation.all_scores)
    

mnist_digits()