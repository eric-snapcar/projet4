# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
def clean( products ):
    thresh = math.ceil(products.shape[0]* 0.6) # au moins 60% du nombre de lignes (arrondi)
    return products.dropna(axis=1,thresh=thresh); # au moins 60% de lignes non nulles
def hist( products_column, range = [0, 100], bins = 50 ):
    products_column.plot(kind='hist',bins = bins, range = range  )
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.show()
    return;
# On charge le dataset

products = pd.read_csv('products.csv', low_memory=False, delimiter='\t', error_bad_lines=False)
products = clean( products )

hist(products['carbohydrates_100g'], [0, 100],50)
#products.hist(column = 'carbohydrates_100g')

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
