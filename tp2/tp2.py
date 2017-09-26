import numpy as np
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
# Le dataset principal qui contient toutes les images
sample = np.random.randint(70000, size=5000)
print sample
data = mnist.data[sample]
target = mnist.target[sample]
print data.shape
print target.shape

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)
