from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
# Le dataset principal qui contient toutes les images
print mnist.data.shape

# Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
print mnist.target.shape
