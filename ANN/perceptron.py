import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# load and extract training data
iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int)

# train perceptron
per_clf = Perceptron(random_state=42, max_iter=5, tol=None)
per_clf.fit(X, y)

# predict based on petal length and width
y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)