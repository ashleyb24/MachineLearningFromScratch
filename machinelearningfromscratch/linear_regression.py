import numpy as np


def _transform_x(x):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return np.insert(x, 0, 1, axis=1)


def _estimate_coefficients(x, y):
    """
        Estimates both the intercept and all coefficients using the ordinary least squares method. \n
        β = (X^T X)^-1 X^T y
    """
    x_transpose = np.transpose(x)
    return np.linalg.inv(x_transpose.dot(x)).dot(x_transpose).dot(y)


class MultipleLinearRegression:

    def __init__(self):
        self.coefficients = []
        self.intercept = 0

    def fit(self, x, y):
        """
            Fits the training data to the model so predictions can later be made.
        """
        x = _transform_x(x)
        betas = _estimate_coefficients(x, y)
        self.intercept = betas[0]
        self.coefficients = betas[1:]

    def predict(self, x):
        """
            Predicts the y value for the given features x using the coefficients calculated in the fit. \n
            y = β_0 + (β_1 * x_1) + (β_2 * x_2) + ... +  (β_n * x_n)
        """
        prediction = self.intercept
        for beta, feature in zip(self.coefficients, x):
            prediction += beta * feature
        return prediction

    def score(self, y_actual, y_prediction):
        """
            Calculate the R squared score for the predicted values. \n
            R^2 = 1 - ∑(y_actual - y_prediction)^2 / ∑(y_actual - y_average)^2
        """
        residual_sum_squares = 0
        total_sum_squares = 0
        for i in range(len(y_actual)):
            residual_sum_squares += (y_actual[i] - y_prediction[i]) ** 2
            total_sum_squares += (y_actual[i] - np.average(y_actual)) ** 2
        return 1 - (residual_sum_squares / total_sum_squares)
