import numpy as np
import pandas as pd
from machinelearningfromscratch.linear_svm import MultiClassLinearSVM
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm


X, y = datasets.make_blobs(
        n_samples=5000, n_features=20, centers=3, cluster_std=1.15, random_state=40
    )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiClassLinearSVM()
model.fit(X_train, y_train)

sklearn_model = svm.SVC()
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)

predictions = []
for row in X_test:
    pred = model.predict(row)
    predictions.append(pred)
predictions = np.array(predictions)

df = pd.DataFrame({
    'Actual': y_test,
    'Prediction': predictions,
    'sklearn Prediction': sklearn_predictions
})

print('model:\n', df)
print('model score:', model.score(predictions, y_test))
print('sklearn score:', sklearn_model.score(X_test, y_test))
