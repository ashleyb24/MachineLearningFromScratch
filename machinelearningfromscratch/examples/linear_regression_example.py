import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from machinelearningfromscratch.linear_regression import MultipleLinearRegression
from sklearn.linear_model import LinearRegression


# use dataset from make_regression
X, y = make_regression(n_samples=100, n_features=20, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# create custom regression model
model = MultipleLinearRegression()
model.fit(X_train, y_train)

# create sklearn regression model
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

sklearn_predictions = sklearn_model.predict(X_test)
predictions = []
for row in X_test:
    pred = model.predict(row)
    predictions.append(pred)

# score the different models
score = model.score(y_test, predictions)
sklearn_score = sklearn_model.score(X_test, y_test)
print("Score: ", score)
print("Score from sklearn model: ", sklearn_score)

df = pd.DataFrame({
    'Actual': y_test,
    'Prediction': predictions,
    'sklearn prediction': sklearn_predictions
})
print("Predictions: \n", df.head())
