import numpy as np


def euclidean_distance(x1, x2):
    """
        d(x1, x2) = ( ∑(x1_i - x2_i)^2 )^0.5
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def _score(predictions, actual):
    return len(predictions[predictions == actual]) / len(predictions)


class KNN:

    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        indices_nearest_neighbours = np.argsort(distances)[:self.k]
        labels_nearest_neighbours = self.y_train[indices_nearest_neighbours]
        label_votes = {}
        for label in labels_nearest_neighbours:
            if label not in label_votes:
                label_votes[label] = 1
            else:
                label_votes[label] += 1
        return max(label_votes, key=label_votes.get)

    def score(self, predictions, actual):
        return _score(predictions, actual)
