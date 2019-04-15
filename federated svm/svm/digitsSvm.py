
import numpy as np

from sklearn import linear_model
from sklearn.datasets import load_digits

digits = load_digits()

print('Shapes:', digits.target.shape)

pivot_point = 1500

digits_learn_data, digits_learn_target, digits_validate_data, digits_validate_target = digits.data[:pivot_point], digits.target[:pivot_point], digits.data[pivot_point:], digits.target[pivot_point:]

number_classifier = linear_model.SGDClassifier(
    alpha=0.001,
    max_iter=100, 
    tol=0.0001,
    random_state=1337,
    shuffle=True,
    warm_start=True,
    )
iteration = 0

#while(input('Press enter for another iteration')):
number_classifier.fit(
    digits_learn_data, 
    digits_learn_target)

iteration += 1
print('Score:', '{:.3}'.format(number_classifier.score(
    digits_validate_data, 
    digits_validate_target)), 'Iterations:', number_classifier.n_iter_)