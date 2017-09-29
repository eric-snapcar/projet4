# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
def clean( products ):
    thresh = math.ceil(products.shape[0]* 0.6) # au moins 60% du nombre de lignes (arrondi)
    return products.dropna(axis=1,thresh=thresh); # au moins 60% de lignes non nulles
# On charge le dataset
products = pd.read_csv('products.csv', delimiter='\t', error_bad_lines=False)
products = clean( products )
print(products.shape)
print(products)
