# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:51:48 2017

@author: Alexandre Truong
"""

#%%
# Importation des librairies nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats
#%%--------------------------Definition of functions------------------------------------------------
def keep_var(products, percent):
    #Affiche le pourcentage de case vide par variable dans le dataframe 'df' 
    #et renvoie une liste de noms de variable qui a un pourcentage d'éléments vides inférieur à 'percent'
    
    # Une liste des éléments à garder
    lis_var_keep=[]
    #une liste des éléments à jeter
    lis_var_bin=[]
    # liste des ratios
    lis_ratio=[]
    # dataframe (2 entrées) pour visualiser les sorties avec à gauche la variable à droite le ratio calculé
    #df_out = pd.DataFrame()
                
    for variable in list(products):
        ratio = products[variable].isnull().sum()/len(products[variable])
        if (ratio < percent):
            lis_var_keep.append(variable)
        else:
            lis_var_bin.append(variable)
        lis_ratio.append(ratio)
        print(variable,'               ',round(ratio,2)) #A CHANGER : SORTIR UN JOLI TABLEAU UTILISER DF OU MATRICE
          
    return lis_var_keep, lis_var_bin, lis_ratio;

#%%
def print_var(products, info = None):
    #Affiche les variables dans le dataframe df, le data type, ainsi que les 5 premiers éléments si 'afficher'='oui'
    
    print('Variables')
    print()
    for var in list(products):
        print('-------------------------------------------------')
        #print(var,' (',df[var].dtype,')','     ',df[var][0],',',df[var][1],',',df[var][2],',',df[var][3],',',df[var][4]) # A améliorer
        print(var,' (',products[var].dtype,')','      ')
        if info == 'oui':
            print(products[var][0:4])
    print('-------------------------------------------------')
    print()
    print('Le nombre de variables est de :',len(list(products)))
    return;
#%%
def hist(products, columnName, xLabel = None, yLabel = 'count',range = [0, 100], bins = 50):
    # A VERSIONNER pour adapter le range à des éléments non numériques
    #Afficher un histogramme en donnant le dataframe et le nom de la variable
    products[columnName].plot(kind='hist',bins = bins, range = range , color = '#C65C66', edgecolor='#F59AA2', )
    plt.xlabel(xLabel or columnName)
    plt.ylabel(yLabel)
    plt.show()
    return;
#%%
def mostFrequent(products,columnName,numberOfValue):
    #renvoie les 'numberOfValue' éléments les plus fréquents d'une variable 'columnName
    return products[columnName].value_counts().head(numberOfValue).index.tolist();
#%%
def density(products, columnName, xLabel = None, yLabel = 'density'):
    #Afficher la densité en donnant le dataframe et le nom de la variable
    density = stats.kde.gaussian_kde(np.array(products[columnName].dropna()))
    x = np.arange(0., 8, .1)
    plt.plot(x, density(x))
    plt.xlabel(xLabel or columnName)
    plt.ylabel(yLabel)
    plt.show()
    return;
#%%
def density_multi_1(products, columnNames, xLabel = None, yLabel = 'density'):
    #Affiche les densités des variables 'columnNames' sur un graphe
    for index, value in enumerate(columnNames):
        density = stats.kde.gaussian_kde(np.array(products[value].dropna()))
        x = np.arange(-15., 30., .1)
        plt.plot(x, density(x), linewidth=1.3,label=value)
    if xLabel is not None:
        plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()
    return;
#%%
def density_multi_2(products, featureName, sampleName, sampleValues, xLabel = None, yLabel = 'density'):
    #Affiche sur un graphe les densités de la variable 'featureName' pour les éléments 'sampleValues' qui sont dans la variable 'sampleName'
    for index, value in enumerate(sampleValues):
        products_ = products[products[sampleName] == value]
        density = stats.kde.gaussian_kde(np.array(products_[featureName].dropna()))
        x = np.arange(0., 100, .1)
        plt.plot(x, density(x), linewidth=1.3,label=value)
    plt.xlabel(xLabel or featureName)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()
    return;
#%%
def scatter(products, columnName1, columnName2, xmin = 0, xmax = 100, ymin = 0, ymax = 100):
    # Pour analyse bi-dim, plot une variable contre une autre, affiche des petits points
    plt.scatter(products[columnName1], products[columnName2], color = '#CB6872', s = 1.325)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(columnName1)
    plt.ylabel(columnName2)
    plt.show()
    return;
#%%
def corr_var(products,columnNames): # A AMELIORER utiliser un dataframe
    #Renvoie toutes les corrélations entre variables dans la liste 'columnNames' du dataframe 'products' dans 
    i=0
    j=0
    for i in range(1,len(columnNames),1):
        for j in range(1,len(columnNames),1):
            if i < j:
                r = products[columnNames[i-1]].dropna().corr(products[columnNames[j-1]].dropna())
                print('Correlation',columnNames[i-1],'/',columnNames[j-1],'       ',round(r,2))   
#%%    

#%%-------------------------------- Importation et Cleaning ------------------------------

#Import des données
nutri_data_raw = pd.read_csv('C:/Users/Atruong1/Documents/Python Scripts/OC/Projet 2/fr.openfoodfacts.org.products.csv', delimiter='\t',error_bad_lines=False) #A modifier
print('Format des données',nutri_data_raw.shape)
# Il y a 320 772 produits pour 162 variables




#%%------------------------------ Sélection des variables à étudier ---------------------
"""
A RAJOUTER L INTRO DU KAGGLE

1 - Etude sur les données

A l'issue de cette partie nous aurons les variables et les données que nous souhaitons étudier.

Il y a en tout 320 772 produits dans notre jeu de données décrites en 162 variables. Celles-ci sont plus ou moins remplies
Pour la suite, je ne vais m'intéresser qu'aux produits français. Après tout Marmiton est un site français. 
On a 94 392 produits français dans notre jeu de données.
"""

# On décide filtrer sur la France (Mamiton site français)
nutri_data_raw_fr = nutri_data_raw[nutri_data_raw['countries_fr']=='France']
#%%
"""
Ces produits sont décrits par les variables suivantes dans notre jeu de données :
"""
# Observons les différentes variables
print_var(nutri_data_raw_fr)

#%%
"""
Beaucoup de variables contiennent énormément de cellules vides. On ne va garder
que les variables dont au plus 40% des lignes ne sont pas remplies de NA afin de travailler avec des variables qui contiennet des informations
de manière significative
"""
percent = 0.4
lis_var_keep, lis_var_bin, lis_ratio = keep_var(nutri_data_raw_fr, percent)

# On filtre notre bdd
nutri_data_int = nutri_data_raw_fr[lis_var_keep]

#%%
"""
Les variables restantes sont :
"""
# Observons ces variables conservées
print_var(nutri_data_int, 'oui')
#%%
"""
On remarque que sur les données restantes, nombreuses sont celles qui n'apporte pas d'information nutritionnelle ou sont en doublon. Nous les enlevons.
Il nous reste finalement les variables suivantes. A noter qu'elles ne seront pas toutes utilisées dans notre étude.

"""
# On observe de nombreuses variable qui ne sont pas utiles pour regarder la nutrition : code, url, creator, created_t, last_modified_t, last_modified_datetime, states
lis_var_useless =['code', 'url', 'creator', 'created_t', 'last_modified_t', 'last_modified_datetime','created_datetime', 'countries','countries_tags','states','states_tags','states_fr','main_category','main_category_fr','categories','categories_tags','categories','categories_fr']

#on filtre et on affiche les varibles restantes à conserver soit 16 variables : elles ne seront pas toutes utilisées
nutri_data = nutri_data_int.drop(lis_var_useless, axis=1)
print_var(nutri_data)




#%%---------------------- Univariate Plots Section -------------------
"""
2 - Analyse univariée

Dans cette partie nous allons faire des études univariées.

Tout d'abord intéresson-nous aux macro-nutriments
"""
#%%
#Macro-nutriments
"""
2.1 - Macro-nutriments
Les macro-nutriments sont composés de lipides, de protéines et de glucides. 
Ils apportent de la matière structurante (acides aminées, lipides) et de l'énergie à l'oganisme.
"""
#Lipides
"""
2.1.1 - Les lipides

Les lipides sont notamment composés de graisse saturée :
"""
hist(nutri_data,'saturated-fat_100g',xLabel='g de graisse saturée pour 100g de produit')

"""
La plupart des produits ne continnent pas de la graisse saturée. La distribution est dissymétrique avec un skew positif cependant
on observe une légère binomialité. Il y  a un tout petit pic autour de 55g de graisse saturée pour 100g de produit.

Les produits contenant de la graisse saturée vers ce pic et au-delà sont :
"""
print(nutri_data[(nutri_data['saturated-fat_100g']>55) & (nutri_data['saturated-fat_100g']<100)]['pnns_groups_1'].dropna())

"""
La plupart de ces produits sont du beurre, huile, sauce.
Il faut éviter de trop consommer ces produtis pour un régime équilibré.

Malheureusement la donnée gramme de graisse pour 100g de produit n'a pas passé le filtre des 40% de NA.
"""
#%%
# Les protéines
"""
2.1.2 - Les protéines
"""
hist(nutri_data,'proteins_100g',xLabel='g de protéines pour 100g de produit')
"""
Distribution dissimétrique avec un skew positif. Il y a peu de produits avec beaucoup de protéines, 
ce sont des compléments alimentaires pour la prise de masse musculaire :
"""
print(nutri_data[(nutri_data['proteins_100g']>70) & (nutri_data['proteins_100g']<100)]['product_name'].dropna())
#%%
# glucides
"""
2.1.3 - Les glucides 

Les glucides sont notamment composés du sucre
"""
hist(nutri_data,'sugars_100g',xLabel='g de sucre pour 100g de produit')
"""
Distribution dissimétrique avec un skew positif. Il y a peu de produits avec beaucoup de protéines, 
Les produits contenant beaucoup de sucre (>80g pour 100g) sont surtout des collations sucrées et des boissons sucrées
"""
print(nutri_data[(nutri_data['sugars_100g']>80) & (nutri_data['sugars_100g']<100)]['pnns_groups_1'].dropna())
#%%
#nutritional score
# A RAJOUTER le double chart un uk l'autre fr
"""
2.2 - Valeur nutritionnelle
2.2.1 Score nutritionnelle

On s'intéresse maintenant au score nutritionnel. Il existe dans nos données un score nutri français et un score nutri uk.

"""
density_multi_1(nutri_data,['nutrition-score-fr_100g','nutrition-score-uk_100g'],xLabel = 'Score nutritionnel')

"""
Peu de différence entre les deux variables. On va conserver le score français par la suite.

Distribution bimodale avec un pic à 0 (cf. ci-dessous : boissons sans sucre, eau, pâtes, nouilles) et un pic entre 10 et 20.

"""
print(nutri_data[(nutri_data['nutrition-score-fr_100g']>-0.5) & (nutri_data['nutrition-score-fr_100g']<0.5)]['product_name'].dropna())


# A RAJOUTER valeeur nutritionnelle
# A RAJOUTER Etudes des Autres Variables


# ??? FAUT-IL FAIRE LES PARTIES OU IL Y A LES QUESTIONS ???
#%%---------------------- Bivariate Plots Section -------------------
"""
Les corrélations entre variables :
"""
# Variables d'intérêts
lis_bivariates=['energy_100g','saturated-fat_100g','sugars_100g','proteins_100g','salt_100g','sodium_100g','nutrition-score-fr_100g','nutrition-score-uk_100g']
# Produire tous les cross scatterplots
# Produire toutes les corrélations
corr_var(nutri_data,lis_bivariates)
#%%
"""
On affiche tous les scatter plots
"""
nb_graph = (len(lis_bivariates)*(len(lis_bivariates)+1)/2)

#%%
scatter(nutri_data,'nutrition-score-fr_100g','saturated-fat_100g')
#%%---------------------- Multivariate Plots Section -------------------
"""
Intéressons nous aux nutriments contenus dans chaque type de produit

Parmi les 5 catégories de nourriture les splus présentes dans nos données, les aliments protéinés contiennent le moins de sucre, a contratrio
les boissons et les collations sucrées le plus (normal)
"""

type_of_products = 'pnns_groups_1'
lis_prod = mostFrequent(nutri_data,type_of_products,6)#modifier pour enlever unknown
density_multi_2(nutri_data,'sugars_100g','pnns_groups_1',lis_prod) 