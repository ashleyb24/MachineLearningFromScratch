import numpy as np
import pandas as pd
from machinelearningfromscratch.knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = KNN()
model.fit(X_train, y_train)

sklearn_model = KNeighborsClassifier(n_neighbors=3)
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

print(df)
print('model score:', model.score(predictions, y_test))
