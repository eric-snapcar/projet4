# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats

def clean( products ):
    thresh = math.ceil(products.shape[0]* 0.6) # au moins 60% du nombre de lignes (arrondi)
    return products.dropna(axis=1,thresh=thresh); # au moins 60% de lignes non nulles
def hist_1( products, columnName, xLabel = None, yLabel = 'count',range = [0, 100], bins = 50 ):
    products[columnName].plot(kind='hist',bins = bins, range = range , color = '#C65C66', edgecolor='#F59AA2' )
    plt.xlabel(xLabel or columnName)
    plt.ylabel(yLabel)
    plt.show()
    return;
def hist_2( products, sampleName, sampleValues, yLabel = 'count', xLabel = None):
    x_spacing = np.arange(len(sampleValues))
    count = products[sampleName].value_counts()
    values = []
    for index, value in enumerate(sampleValues):
        values.append(count[value])
    plt.bar(x_spacing, values, align='center', color = '#C65C66', edgecolor='#C65C66' )
    plt.xticks(x_spacing, sampleValues)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel or sampleName)
    plt.show()
    return;
def density( products, featureName, xLabel = None, yLabel = 'density'):
    density = stats.kde.gaussian_kde(np.array(products[columnName].dropna()))
    x = np.arange(0., 8, .1)
    plt.plot(x, density(x),color='#060606',linewidth=1.3)
    plt.fill_between(x,density(x),color='#9AD8DA', alpha=.45)
    plt.xlabel(xLabel or columnName)
    plt.ylabel(yLabel)
    plt.show()
    return;
def density_multi( products, columnNames, xLabel = None, yLabel = 'density'):
    for index, value in enumerate(columnNames):
        density = stats.kde.gaussian_kde(np.array(products[value].dropna()))
        x = np.arange(0., 8, .1)
        plt.plot(x, density(x), linewidth=1.3,label=value)
    if xLabel is not None:
        plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()
    return;
def density_multi_1( products, columnNames, xLabel = None, yLabel = 'density'):
    for index, value in enumerate(columnNames):
        density = stats.kde.gaussian_kde(np.array(products[value].dropna()))
        x = np.arange(0., 8, .1)
        plt.plot(x, density(x), linewidth=1.3,label=value)
    if xLabel is not None:
        plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()
    return;
def density_multi_2( products, featureName, sampleName, sampleValues, xLabel = None, yLabel = 'density'):
    for index, value in enumerate(sampleValues):
        products_ = products[products[sampleName] == value]
        density = stats.kde.gaussian_kde(np.array(products_[featureName].dropna()))
        x = np.arange(0., 8, .1)
        plt.plot(x, density(x), linewidth=1.3,label=value)
    plt.xlabel(xLabel or featureName)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()
    return;
def scatter( products, column1, column2, xmin = 0, xmax = 100, ymin = 0, ymax = 100 ):
    plt.scatter(products[column1], products[column2], color = '#CB6872', s = 1.325)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()
    return;
def mostFrequent( products,columnName,numberOfValue):
    return products[columnName].value_counts().head(numberOfValue).index.tolist();

# read_csv
products = pd.read_csv('products.csv', low_memory=False, delimiter='\t', error_bad_lines=False)

# clean
products = clean( products )

# hist_1
# hist_1(products,'nutrition-score-uk_100g')

# density
#density(products,'nutrition-score-uk_100g')

# density_multi_1
# density_multi_1(products,['nutrition-score-uk_100g','nutrition-score-fr_100g'],'nutrition-score-uk-fr_100g')

# scatter
# scatter(products,'fat_100g','saturated-fat_100g')
# scatter(products,'nutrition-score-fr_100g','saturated-fat_100g',xmax=50)

# mostFrequent
# print mostFrequent(products,'brands',5)
# print mostFrequent(products,'countries',5)

# density_multi_2
# sampleName = 'brands'
# sampleValues = mostFrequent(products,sampleName,5)
# density_multi_2(products,'sugars_100g',sampleName,sampleValues)
# density_multi_2(products,'nutrition-score-fr_100g',sampleName,sampleValues)

# hist_2
sampleName = 'brands'
sampleValues = mostFrequent(products,sampleName,5)
hist_2( products,  sampleName, sampleValues )

""" column values
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
