#print(__name__)

import numpy as np

from copy import deepcopy

from sklearn import datasets
from sklearn import linear_model

from generator import make_class_dataset, make_class_dataset_test


class SVM:
    def __init__(self, X=None, y=None, seed=1337):
        self.X = X
        self.y = y
        self.clf = linear_model.SGDClassifier(
            alpha=0.001, 
            max_iter=1, 
            tol=0.0001, 
            random_state=seed,
            warm_start=True,
            shuffle=True,
            )

    def load_dataset(self):
        self.X, self.y, self.X_test, self.y_test = make_class_dataset_test()

    def partial_fit(self, coef=None, intercept=None):
        self.clf.fit(self.X, self.y, coef_init=coef, intercept_init=intercept)

    def get_score(self, X_test=None, y_test=None):
        if (X_test is None and y_test is None):
            return self.clf.score(self.X_test, self.y_test)
        else:
            return self.clf.score(X_test, y_test)

class Federated_SVM:
    def __init__(self, X, y, X_test, y_test):
        self.X = X
        self.y = y 
        self.X_test = X_test
        self.y_test = y_test
        
    # Needs a complete SGDclassifier already set up before
    def generate_federation(self, SGDclassifier, amount = 4):

        self.federation = []
        
        for i in range(amount):
            self.federation.append(deepcopy(SGDclassifier))
    
    def fit_federation(self, iterations = 1, datasize = 1000):

        for _ in range(iterations):

            for idx, SVM in enumerate(self.federation):
                start_point = idx * datasize
                end_point = (idx+1) * datasize
                SVM.fit(self.X[start_point:end_point], self.y[start_point:end_point])
