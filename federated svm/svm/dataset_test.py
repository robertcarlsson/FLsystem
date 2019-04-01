import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model

n_features = 2

samples = 200
test_ratio = 0.2 
test_index = int(test_ratio * samples)

X, y = datasets.make_classification(
    n_samples = samples,
    n_features = n_features,
    n_informative = 2,
    n_redundant = 0,
    n_classes = 3,
    n_clusters_per_class = 1,
    random_state = 1
)

X_test = X[test_index:]
y_test = y[test_index:]
X = X[:test_index]
y = y[:test_index]


# Step size in the mesh
h = .02 

clf = linear_model.SGDClassifier(alpha=0.001, max_iter=1, tol=0.0001)#.fit(X[:, :2],y)

for i in range(3):

    clf = clf.partial_fit(X[:, :2],y, np.unique(y))

    # create a mesh to plot in

    features_mesh = []

    #for i in range(n_features):


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))


    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('tight')

    colors = 'brg'

    labels = ['att1', 'att2']

    # Plot also the training points
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label='class' + str(i+1),
                    cmap=plt.cm.Paired, edgecolor='black', s=20)
    plt.title("Decision surface of multi-class SGD")
    plt.axis('tight')

    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = clf.coef_
    intercept = clf.intercept_

    print("Model:")

    score = clf.score(X_test, y_test)
    

    with np.printoptions(precision=2):
        print('Score: ', score)
        print('Coef: \t', *coef)
        print('Intercept: ', intercept)

    # Shuffle
    idx = np.arange(X_test.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X_show = X_test[idx]
    y_show = y_test[idx]
    X_show = X_show[:100]
    y_show = y_show[:100]

    colors = ['deepskyblue', 'lightcoral', 'limegreen']

    for i, color in zip(clf.classes_, colors):
        idx = np.where(y_show == i)
        plt.scatter(X_show[idx, 0], X_show[idx, 1], c=color, label='predictions' + str(i+1),
                    cmap=plt.cm.Paired, edgecolor='black', s=40)

    #print(coef)
    #print(intercept)


    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

        plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                ls="--", color=color)


    #for i, color in zip(clf.classes_, colors):
    #    plot_hyperplane(i, color)
    plt.legend()
    plt.show()

