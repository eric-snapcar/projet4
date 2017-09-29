# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats

def clean( products ):
    thresh = math.ceil(products.shape[0]* 0.6) # au moins 60% du nombre de lignes (arrondi)
    return products.dropna(axis=1,thresh=thresh); # au moins 60% de lignes non nulles
def hist( products, columnName, xLabel = None, yLabel = 'count',range = [0, 100], bins = 50 ):
    products[columnName].plot(kind='hist',bins = bins, range = range , color = '#C65C66', edgecolor='#F59AA2', )
    plt.xlabel(xLabel or columnName)
    plt.ylabel(yLabel)
    plt.show()
    return;
def density( products, columnName, xLabel = None, yLabel = 'density'):
    density = stats.kde.gaussian_kde(np.array(products[columnName].dropna()))
    x = np.arange(0., 8, .1)
    plt.plot(x, density(x))
    plt.xlabel(xLabel or columnName)
    plt.ylabel(yLabel)
    plt.show()
    return;
# On charge le dataset

products = pd.read_csv('products.csv', low_memory=False, delimiter='\t', error_bad_lines=False)
products = clean( products )
# Plot de carbohydrates_100g
# hist(products,'nutrition-score-uk_100g')


density(products,'nutrition-score-uk_100g')

"""
density = stats.kde.gaussian_kde(np.array(products['nutrition-score-uk_100g'].dropna()))
x = np.arange(0., 8, .1)
plt.plot(x, density(x))
plt.show()
"""
"""
import seaborn as sns

sns.set_style('whitegrid')
sns.kdeplot(np.array(products['nutrition-score-uk_100g']), bw=0.5)
"""
"""
print(products.columns.values)
['code' 'url' 'creator' 'created_t' 'created_datetime' 'last_modified_t'
 'last_modified_datetime' 'product_name' 'brands' 'brands_tags' 'countries'
 'countries_tags' 'countries_fr' 'ingredients_text' 'serving_size'
 'additives_n' 'additives' 'ingredients_from_palm_oil_n'
 'ingredients_that_may_be_from_palm_oil_n' 'nutrition_grade_fr' 'states'
 'states_tags' 'states_fr' 'energy_100g' 'fat_100g' 'saturated-fat_100g'
 'carbohydrates_100g' 'sugars_100g' 'fiber_100g' 'proteins_100g'
 'salt_100g' 'sodium_100g' 'nutrition-score-fr_100g'
 'nutrition-score-uk_100g']
 """
