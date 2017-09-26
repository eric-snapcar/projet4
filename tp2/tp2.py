from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
# Le dataset principal qui contient toutes les images
print mnist.data.shape
print mnist.target
