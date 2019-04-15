
import numpy as np

from sklearn import linear_model
from sklearn.datasets import load_digits

from svm import Federated_SVM, SVM

digits = load_digits()

print('Shapes:', digits.target.shape)

pivot_point = 1500

digits_learn_data, digits_learn_target, digits_validate_data, digits_validate_target = digits.data[:pivot_point], digits.target[:pivot_point], digits.data[pivot_point:], digits.target[pivot_point:]

n_svm = 4

X_list = np.split(digits_learn_data[:1000], n_svm)
y_list = np.split(digits_learn_target[:1000], n_svm)

federation = Federated_SVM(digits_validate_data, digits_validate_target)

for i in range(n_svm):
    federation.add_participant(SVM(X_list[i], y_list[i]))

federation.run_eon(50)

print(federation.all_scores)

'''
number_classifier = linear_model.SGDClassifier(
    alpha=0.001,
    max_iter=1, 
    tol=0.001,
    random_state=1339,
    shuffle=True,
    warm_start=True,
    )
iteration = 0
number_classifier.fit(
        digits_learn_data, 
        digits_learn_target)

while(iteration < 4):
    number_classifier.partial_fit(
        digits_learn_data, 
        digits_learn_target)
        #number_classifier.coef_,
        #number_classifier.intercept_)
    iteration += 1

print('Score:', '{:.3}'.format(number_classifier.score(
    digits_validate_data, 
    digits_validate_target)), 'Iterations:', number_classifier.n_iter_, iteration)
'''