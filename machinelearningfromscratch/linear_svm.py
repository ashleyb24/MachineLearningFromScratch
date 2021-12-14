import numpy as np
import copy


def _score(predictions, actual):
    return len(predictions[predictions == actual]) / len(predictions)


class LinearSVM:

    def __init__(self, learning_rate=0.001, iterations=1000, regularisation_param=10000):
        self._weights = None
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._regularisation_param = regularisation_param

    def _calculate_cost_gradients(self, weights, X, y):
        distances = 1 - (y * np.dot(X, weights))
        dw = np.zeros(len(weights))
        for index, d in enumerate(distances):
            if max(0, d) == 0:
                dw += weights
            else:
                dw += weights - (self._regularisation_param * y[index] * X[index])
        return dw/len(y)

    def fit(self, X, y):
        weights = np.zeros(np.shape(X[1]))
        for _ in range(self._iterations):
            gradient = self._calculate_cost_gradients(weights, X, y)
            weights -= self._learning_rate * gradient
        self._weights = weights

    def predict(self, x, sign=True):
        prediction = np.dot(x, self._weights)
        if sign:
            return np.sign(prediction)
        else:
            return prediction

    def score(self, predictions, actual):
        return _score(predictions, actual)


class MultiClassLinearSVM:

    def __init__(self):
        self.linear_svm = LinearSVM()
        self.classifiers = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for classification in classes:
            rest = classes[classes != classification]
            y_copy = copy.copy(y)
            y_copy = np.where(np.isin(y_copy, rest), -1, 1)
            model = copy.copy(self.linear_svm)
            model.fit(X, y_copy)
            self.classifiers[classification] = model

    def predict(self, x):
        predictions = {}
        for classification, model in self.classifiers.items():
            prediction = model.predict(x, sign=False)
            predictions[classification] = prediction
        return max(predictions, key=lambda x: predictions[x])

    def score(self, predictions, actual):
        return _score(predictions, actual)
