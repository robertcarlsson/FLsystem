
from sklearn import datasets



def make_class_dataset_test(samples=200, test_ratio=0.2, n_features=2, random_seed=1):

    test_index = samples - int(test_ratio * samples)

    X, y = datasets.make_classification(
        n_samples = samples,
        n_features = n_features,
        n_informative = 2,
        n_redundant = 0,
        n_classes = 3,
        n_clusters_per_class = 1,
        random_state = random_seed
    )

    X_test = X[test_index:]
    y_test = y[test_index:]
    X = X[:test_index]
    y = y[:test_index]

    #print('Test index:', test_index)

    return X, y, X_test, y_test

def make_class_dataset(samples=200, n_features=2, random_seed=1):

    X, y = datasets.make_classification(
        n_samples = samples,
        n_features = n_features,
        n_informative = 2,
        n_redundant = 0,
        n_classes = 3,
        n_clusters_per_class = 1,
        random_state = random_seed
    )

    return X, y

def print_datasets(arrays):
    for array in arrays:
        print(array.shape)

#print_datasets(make_class_dataset())