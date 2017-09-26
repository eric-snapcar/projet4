import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
mnist = fetch_mldata('MNIST original')
# Le dataset principal qui contient toutes les images
sample = np.random.randint(70000, size=5000)
print sample
data = mnist.data[sample]
target = mnist.target[sample]
print data.shape
print target.shape

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)
error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)

errors = []
for k in range(2,7):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
plt.plot(range(2,7), errors, 'o-')
plt.show()
