import matplotlib.pyplot as plt
import plot_utils
from machinelearningfromscratch.knn import KNN
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

model = KNN(k=10)
model.fit(X, y)

# plot scatter
fig, ax = plt.subplots()
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = plot_utils.make_meshgrid(X0, X1)

plot_utils.plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

plt.show()
