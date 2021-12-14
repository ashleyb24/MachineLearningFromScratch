import numpy as np
import matplotlib.pyplot as plt
from machinelearningfromscratch.linear_svm import MultiClassLinearSVM
from sklearn import datasets


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    predictions = []
    for features in np.c_[xx.ravel(), yy.ravel()]:
        pred = clf.predict(features)
        predictions.append(pred)
    Z = np.array(predictions).reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


X, y = datasets.make_blobs(
        n_samples=500, n_features=10, centers=3, cluster_std=1.15, random_state=40
    )
X_subset = X[:, [5, 7]]  # take subset of features to plot on 2d graph
model = MultiClassLinearSVM()
model.fit(X_subset, y)

# plot scatter
fig, ax = plt.subplots()
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_subset[:, 0], X_subset[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

plt.show()

