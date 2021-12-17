import matplotlib.pyplot as plt
import plot_utils
from machinelearningfromscratch.linear_svm import MultiClassLinearSVM
from sklearn import datasets


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
xx, yy = plot_utils.make_meshgrid(X0, X1)

plot_utils.plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

plt.show()
