#print(__name__)

import numpy as np

from sklearn import datasets
from sklearn import linear_model

from generator import make_class_dataset, make_class_dataset_test


class SVM:
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y
        self.clf = linear_model.SGDClassifier(
            alpha=0.001, 
            max_iter=1, 
            tol=0.0001, 
            warm_start=True,
            shuffle=False,
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


