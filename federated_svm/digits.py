import sys
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.datasets import load_digits

from svm import Federated_SVM, SVM

FEDAVG = 1
FEDHIG = 2
FEDRAN = 3

print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))

if (len(sys.argv) != 3):
    print ("This file needs these arguments:\n")
    print ("n_datapoints \t\tNumber of datapoints \nn_device \t\tNumber of devices \nn_rounds \t\tNumber of rounds \nshared_init \t\tInitialization option, \"no\" or \"yes\" \n")
    exit()


def scikit_digits():

    digits = load_digits()
    print('Shapes:', digits.target.shape)

    pivot_point = 1500

    digits_learn_data, digits_learn_target, \
        digits_validate_data, digits_validate_target = \
        digits.data[:pivot_point], digits.target[:pivot_point], \
        digits.data[pivot_point:], digits.target[pivot_point:]

    mean = digits_learn_data.mean(axis=0)
    std = digits_learn_data.std(axis=0)
    #mean = np.around(mean, decimals=2)
    #std = np.around(std, decimals=2)
    print('Mean:', mean, 'Std:', std)
    std += 0.05  # [val for val in std if v]
    #digits_learn_data = (digits_learn_data - mean) / std

    n_svm = 4

    X_list = np.split(digits_learn_data[:1500], n_svm)
    y_list = np.split(digits_learn_target[:1500], n_svm)

    federation = Federated_SVM(
        digits_validate_data,
        digits_validate_target,
        global_aggregation=True)

    for i in range(n_svm):
        federation.add_participant(SVM(X_list[i], y_list[i], 5334, 1))

    federation.run_eon(15)

    print(federation.all_scores)
    # print(digits_learn_data[0])
    # print(y_list[1][:100])


def mnist_digits(number_of_devices=4, data_points=24000, algorithm=FEDAVG, balanced_dataset=True, num_eons=13):

    folder = '.\\federated_svm\dataset\MNIST\\'
    folder = './federated_svm/dataset/MNIST/'
    X = idx2numpy.convert_from_file(folder + 'train-images.idx3-ubyte')
    y = idx2numpy.convert_from_file(folder + 'train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file(folder + 't10k-images.idx3-ubyte')
    y_test = idx2numpy.convert_from_file(folder + 't10k-labels.idx1-ubyte')

    X = np.array([elm.ravel() for elm in X])
    X_test = np.array([elm.ravel() for elm in X_test])

    n_svm = number_of_devices
    n_datapoints = data_points

    #X = X[:n_datapoints]
    #y = y[:n_datapoints]

    # standardize
    #mean = X[:200].mean(axis=0)
    #std = X[:200].std(axis=0)
    #print('Mean:', mean, 'Std:', std)
    #std += 0.01
    #X = (X - mean) / std
    X_list = np.split(X[:n_datapoints], n_svm)
    y_list = np.split(y[:n_datapoints], n_svm)

    if not balanced_dataset:
        X_list = [X_list[0][:40], X_list[1][:200],
                  X_list[2][:1000], X_list[3][:6000]]
        y_list = [y_list[0][:40], y_list[1][:200],
                  y_list[2][:1000], y_list[3][:6000]]

    federation = Federated_SVM(
        X[n_datapoints:n_datapoints+1000],
        y[n_datapoints:n_datapoints+1000],
        X_test,
        y_test,
        global_aggregation=False,
        algorithm=algorithm,
        tol=0.0001)

    for i in range(n_svm):
        federation.add_participant(SVM(X_list[i], y_list[i], 5333, 1))

    #num_eons = eons
    federation.run_eon(num_eons)

    algorithm_string = [
        "FederatedAveraging",
        "FederatedHighest",
        "FederatedRandom"
    ]
    balanced_string = [
        "Balanced dataset",
        "Unbalanced dataset"
    ]

    b_i = 0
    if not balanced_dataset:
        b_i = 1

    top_label = algorithm_string[algorithm-1] + " - " + balanced_string[b_i]
    # print(y_list[1][:100])
    print(federation.all_scores)
    conv_iter = federation.global_model_scores.index(
        max(federation.global_model_scores))
    print('Number of iterations until convergence:', conv_iter)
    print(federation.global_model_scores)
    # print(federation.federation[0].clf.n_iter)
    # print(X[:3][0])
    plt.title(top_label)
    plt.plot(federation.all_scores)
    plt.axis([0, num_eons-1, 0.5, 1.0, ])

    labels = []
    for n in range(n_svm):
        labels.append('Device' + str(n+1))

    #labels.append('Global Model')

    plt.legend(labels)

    plt.show()

# scikit_digits()

if sys.argv[2] == 'y':
    balanced_dataset = True
else:
    balanced_dataset = False

mnist_digits(number_of_devices=4, data_points=24000,
             algorithm=int(sys.argv[1]), balanced_dataset=balanced_dataset,
             num_eons=13)
