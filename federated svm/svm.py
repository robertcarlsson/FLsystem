#print(__name__)

import numpy as np

from copy import deepcopy

from sklearn import datasets
from sklearn import linear_model

from test.generator import make_class_dataset, make_class_dataset_test


class SVM:
    def __init__(self, X=None, y=None, seed=1338, n_iter=1):
        self.X = X
        self.y = y
        self.clf = linear_model.SGDClassifier(
            alpha=0.0001, 
            max_iter=1, 
            tol=0.001, 
            random_state=seed,
            shuffle=True,
            warm_start=False,
            early_stopping=False,
            n_iter_no_change=5,
            learning_rate='adaptive',
            eta0=0.01
            )
        self.n_iter = n_iter
        self.score = 0

    def __str__(self):
        return 'SVM machine'

    def __repr__(self):
        return 'Represent SVM machine'

    def load_dataset(self):
        self.X, self.y, self.X_test, self.y_test = make_class_dataset_test()

    def partial_fit(self):
        self.clf.partial_fit(self.X, self.y)

    def initial_fit(self, coef=None, intercept=None):
        self.clf.fit(self.X, self.y, coef_init=coef, intercept_init=intercept)

    def run_SGD_iterations(self):
        counter = 1
        while(counter < self.n_iter):
            self.partial_fit()
            counter += 1

    def get_score(self, X_test=None, y_test=None):
        if (X_test is None and y_test is None):
            return self.clf.score(self.X_test, self.y_test)
        else:
            return self.clf.score(X_test, y_test)

class Federated_SVM:
    def __init__(self, X, y, X_test, y_test, 
    epoch_iterations=1, global_aggregation=True, tol=0.01):
        # Static test suite so it is the same for all evaluations
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.epoch_iterations = epoch_iterations
        self.global_aggregation = global_aggregation

        self.global_coef_ = None
        self.global_intercept_ = None

        self.federation = np.array([])
        self.all_scores = None
        
        self.tol = tol
        self.global_model_scores = []
        self.n_iterations = 0
        self.global_clf = linear_model.SGDClassifier(
            alpha=0.0001, 
            max_iter=1, 
            tol=0.001, 
            random_state=1,
            shuffle=True,
            warm_start=False,
            early_stopping=True,
            n_iter_no_change=5,
            learning_rate='adaptive',
            eta0=0.01
            )

    def __str__(self):
        return 'Federation quality not yet evaluated'

    def add_participant(self, participant):
        self.federation = np.append(self.federation, participant)

    def _aggregate_highest(self):
        #: Go through the scores to select the model with
        #: highest accuracy

        max_index = np.argmax(self.all_scores[self.n_iterations])

        temp_coef = self.federation[max_index].clf.coef_
        temp_intercept = self.federation[max_index].clf.intercept_

        #: Set an initial value if global model is None
        if self.global_coef_ is None and self.global_intercept_ is None:
            self.global_coef_ = temp_coef
            self.global_intercept_ = temp_intercept
        else:
            self.global_coef_ = (self.global_coef_ + temp_coef) / 2
            self.global_intercept_ = (self.global_intercept_ + temp_intercept) / 2


    def _aggregate_models(self):
        temp_coef = self.federation[0].clf.coef_
        temp_intercept = self.federation[0].clf.intercept_
        
        #: Calculate the federated avarage and update the global model
        for svm in self.federation[1:]:
            temp_coef += svm.clf.coef_
            temp_intercept += svm.clf.intercept_
        
        temp_coef /= len(self.federation)
        temp_intercept /= len(self.federation)

        #: Set an initial value if global model is None
        if self.global_coef_ is None and self.global_intercept_ is None:
            self.global_coef_ = temp_coef
            self.global_intercept_ = temp_intercept
        else:
            self.global_coef_ = (self.global_coef_ + temp_coef) / 2
            self.global_intercept_ = (self.global_intercept_ + temp_intercept) / 2
        
    def _update_global_score(self):
        self.global_clf.fit(
            X=self.X, 
            y=self.y, 
            coef_init=self.global_coef_,
            intercept_init=self.global_intercept_)
        self.n_iterations += 1
        self.global_model_scores.append(
            self.global_clf.score(
                self.X_test,
                self.y_test))

    def run_epoch(self):
         
        epoch_local_scores = np.array([])
        
        #: Initialize the run for the participants
        #: Then run the SGD iterations for n times
        #: get the local accuracy and send to server
        for svm in self.federation:
            svm.initial_fit(
                coef=self.global_coef_,
                intercept=self.global_intercept_
            )
            svm.run_SGD_iterations()
            epoch_local_scores  = np.append(epoch_local_scores, svm.get_score(self.X_test, self.y_test))

        #: Collect the scores for plotting
        if self.all_scores is None:
            self.all_scores = np.array([epoch_local_scores])
        else:
            self.all_scores = np.append(self.all_scores, np.array([epoch_local_scores]), axis=0)

        #: Create the aggregated global model
        if self.global_aggregation:
            self._aggregate_models()
        else:
            self._aggregate_highest()

        self._update_global_score()

        

    def score_global_model(self):
        return self.global_clf.score(self.X_test, self.y_test)

    def run_eon(self, n_epochs=1, min_epochs=15):
        while n_epochs > 0:
            self.run_epoch()
            #print ()
            if len(self.global_model_scores) > 2:
                if self.global_model_scores[-1] < (self.global_model_scores[-2] + self.tol) and n_epochs < min_epochs:
                    n_ep = 0
            n_epochs -= 1
    
