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
from scipy import stats
import seaborn as sns


#%%--------------------------Definition of functions------------------------------------------------
def keep_var(products, percent):
    #Affiche le pourcentage de case vide par variable dans le dataframe 'df' 
    #et renvoie une liste de noms de variable qui a un pourcentage d'éléments vides inférieur à 'percent'
    
    # Une liste des éléments à garder
    lis_var_keep = []
    #une liste des éléments à jeter
    lis_var_bin = []
    # liste des ratios
    #col = ['variable', 'ratio']
    lis_ratio = []
    # dataframe (2 entrées) pour visualiser les sorties avec à gauche la variable à droite le ratio calculé
    #df_out = pd.DataFrame()
                
    for variable in list(products):
        ratio = products[variable].isnull().sum()/len(products[variable])
        if (ratio < percent):
            lis_var_keep.append(variable)
        else:
            lis_var_bin.append(variable)
        #new_line = pd.Series([variable, ratio], index = col)
        lis_ratio.append([variable,ratio])
        print(variable,'               ',round(ratio,2)) #A CHANGER : SORTIR UN JOLI TABLEAU UTILISER DF OU MATRICE
          
    return lis_var_keep, lis_var_bin, lis_ratio;

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

def hist_1(products, columnName, xLabel = None, yLabel = 'count',range = [0, 100], bins = 50 , save = False):
    products[columnName].plot(kind='hist',bins = bins, range = range , color = '#C65C66', edgecolor='#F59AA2' )
    plt.xlabel(xLabel or columnName)
    plt.ylabel(yLabel)
    if save:
        plt.savefig('hist_{}.png'.format(columnName))
        plt.close()
    else:
        plt.show()
    return;

def hist_2(products, sampleName, sampleValues, yLabel = 'count', xLabel = None , save = False):
    x_spacing = np.arange(len(sampleValues))
    values_count = products[sampleName].value_counts()
    values = []
    for index, value in enumerate(sampleValues):
        values.append(values_count[value])
    plt.bar(x_spacing, values, align='center', color = '#C65C66', edgecolor='#C65C66' )
    plt.xticks(x_spacing, sampleValues, rotation=90)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel or sampleName)
    if save:
        plt.savefig('box_{}.png'.format(sampleName))
        plt.close()
    else:
        plt.show()
    return;

def mostFrequent(products,columnName,numberOfValue):
    #renvoie les 'numberOfValue' éléments les plus fréquents d'une variable 'columnName
    return products[columnName].value_counts().head(numberOfValue).index.tolist();

def density(products, columnName, xLabel = None, yLabel = 'density',save = False):
    #Afficher la densité en donnant le dataframe et le nom de la variable
    density = stats.kde.gaussian_kde(np.array(products[columnName].dropna()))
    x = np.arange(0., 8, .1)
    plt.plot(x, density(x))
    plt.xlabel(xLabel or columnName)
    plt.ylabel(yLabel)
    if save:
        plt.savefig('density_{}.png'.format(columnName))
        plt.close()
    else:
        plt.show()
    return;

def density_multi_1(products, columnNames,xmin=-15,xmax=30, xLabel = None, yLabel = 'density',save = False):
    #Affiche les densités des variables 'columnNames' sur un graphe
    for index, value in enumerate(columnNames):
        density = stats.kde.gaussian_kde(np.array(products[value].dropna()))
        x = np.arange(xmin,xmax, .1)
        plt.plot(x, density(x), linewidth=1.3,label=value)
    if xLabel is not None:
        plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    if save:
        plt.savefig('density_{}.png'.format('_'.join(columnNames)))
        plt.close()
    else:
        plt.show()
    return;

def density_multi_2(products, featureName, sampleName, sampleValues, xmin=0,xmax=100, xLabel = None, yLabel = 'density', save=False):
    #Affiche sur un graphe les densités de la variable 'featureName' pour les éléments 'sampleValues' qui sont dans la variable 'sampleName'
    for index, value in enumerate(sampleValues):
        products_ = products[products[sampleName] == value]
        density = stats.kde.gaussian_kde(np.array(products_[featureName].dropna()))
        x = np.arange(xmin,xmax, .1)
        plt.plot(x, density(x), linewidth=1.3,label=value)
    plt.xlabel(xLabel or featureName)
    plt.ylabel(yLabel)
    plt.legend()
    if save:
        plt.savefig('density_{}_{}.png'.format(featureName,sampleName))
        plt.close()
    else:
        plt.show()
    return;

def scatter(products, columnName1, columnName2, xmin = 0, xmax = 100, ymin = 0, ymax = 100, save=False):
    # Pour analyse bi-dim, plot une variable contre une autre, affiche des petits points
    plt.scatter(products[columnName1], products[columnName2], color = '#CB6872', s = 10)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(columnName1)
    plt.ylabel(columnName2)
    if save:
        plt.savefig('scatter_{}_{}.png'.format(columnName1,columnName2))
        plt.close()
    else:
        plt.show()
    return;

def super_scatter(products,columnNames,nb_col,save=False):
    # plot une matrice de scatterplot avec toutes combinaison possible dans 'columnNames" dans un format avec 'nb_col' colonnes
    i=0
    j=0
    k=0
    
    nb_total = len(columnNames)*(len(columnNames)-1)/2
    
    f, ax = plt.subplots(int(nb_total/nb_col),nb_col)
    
    while (i+j) < len(columnNames):
        for i in range(0,len(columnNames),1):
            for j in range(i+1,len(columnNames),1):
                axis = ax[int(k/3),k%nb_col]
                
                axis.scatter(products[columnNames[i]], products[columnNames[j]], color = '#CB6872', s = 3)
                #axarr[int(k/3),k%nb_col].xlim(xmin, xmax)
                #axarr[int(k/3),k%nb_col].ylim(ymin, ymax)
                axis.autoscale()
                #axis.set_axis_off()
                axis.set_xlabel(columnNames[i],labelpad=0.5,size=6)
                axis.set_ylabel(columnNames[j],labelpad=0.5,size=6)
                axis.set_xticklabels([])
                axis.set_yticklabels([])
                #.set_aspect('equal')
                k=k+1
                
    f.subplots_adjust(wspace=0.3, hspace=0.8)
    
    if save:
        plt.savefig('super_scatter_{}.png'.format(columnNames))
        plt.close()
    else:
        plt.show()
    return;
            
                #ax[int(k/3),k%nb_col].show()
                
    return;

def corr_var(products,columnNames): # A AMELIORER utiliser un dataframe
    #Renvoie toutes les corrélations entre variables dans la liste 'columnNames' du dataframe 'products' dans 
    i=0
    j=0
    for i in range(1,len(columnNames),1):
        for j in range(1,len(columnNames),1):
            if i < j:
                r = products[columnNames[i-1]].dropna().corr(products[columnNames[j-1]].dropna())
                print('Correlation',columnNames[i-1],'/',columnNames[j-1],'       ',round(r,2))   
    
def boxplot_multi(products,featureName, sampleName, sampleValues, title=False, save=False):
    # Plot un boxplot de la feature name avec en abscisse les Sample Values sélectionné dans la variable Samplena
    products = products[products[sampleName].isin(sampleValues)]
    products.boxplot(column=[featureName], by=[sampleName])
    plt.title(title)
    if save:
        plt.savefig('boxplot_multi_{}_{].png'.format(featureName,sampleName))
        plt.close()
    else:
        plt.show()
    return;

#def super_boxplot(products,featureName, featureValues, sampleName,sampleValues,nb_col):
#    # plot une matrice de scatterplot avec toutes combinaison possible dans 'columnNames" dans un format avec 'nb_col' colonnes
#            
#    f, ax = plt.subplots(int(len(featureValues)/nb_col),nb_col)
#    k=0
#    
#    for feat in featureValues:
#           axis = ax[int(k/3),k%nb_col]
#           prod = products[(products[featureName]==feat) & (products[sampleName].isin(sampleValues))][sampleName]
#           axis.boxplot(prod)
#                    
#           #axarr[int(k/3),k%nb_col].xlim(xmin, xmax)
#           #axarr[int(k/3),k%nb_col].ylim(ymin, ymax)
#           axis.autoscale()
#           #axis.set_axis_off()
#           axis.set_title(feat,labelpad=0.5,size=7)
#           #axis.set_ylabel(columnNames[j],labelpad=0.5,size=6)
#           axis.set_xticklabels([])
#           axis.set_yticklabels([])
#           #.set_aspect('equal')
#           k=k+1
#               
#    f.subplots_adjust(wspace=0.3, hspace=0.8)
#            
#                #ax[int(k/3),k%nb_col].show()
#                
#    return;


def filter_spe(products, columnName, boolProducts):
    #Affiche les éléments de columnName de products répondant à contdition boolProducts
    print(products[boolProducts][columnName].dropna())

#%%-------------------------------- Importation et Cleaning ------------------------------

#Import des données
nutri_data_raw = pd.read_csv('C:/Users/Atruong1/Documents/Python Scripts/OC/Projet 2/fr.openfoodfacts.org.products.csv', delimiter='\t',error_bad_lines=False) #A modifier
print('Format des données',nutri_data_raw.shape)
# Il y a 320 772 produits pour 162 variables



#%%------------------------------ Sélection des variables à étudier ---------------------
"""
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
que les variables dont au plus 60% des lignes ne sont pas remplies de NA afin de travailler avec des variables qui contiennet des informations
de manière significative
"""
percent = 0.6
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
lis_var_useless =['code', 'url', 'creator', 'created_t', 'last_modified_t', 'last_modified_datetime','created_datetime', 'quantity', 'countries','countries_tags','states','states_tags','states_fr','main_category','main_category_fr','categories','categories_tags','categories','categories_fr','packaging','packaging_tags','purchase_places','ingredients_text','additives_n','additives','ingredients_from_palm_oil_n','ingredients_that_may_be_from_palm_oil_n','image_url','image_small_url']

#on filtre et on affiche les varibles restantes à conserver soit 16 variables : elles ne seront pas toutes utilisées
nutri_data = nutri_data_int.drop(lis_var_useless, axis=1)
print_var(nutri_data)

#%%


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

Les lipides sont notamment composés de graisse saturée (utilisée dans la formule du nutritional score) - maladie cardio-vasculaire :
"""
hist_1(nutri_data,'saturated-fat_100g',xLabel='g de graisse saturée pour 100g de produit')

"""
La plupart des produits ne continnent pas de la graisse saturée. La distribution est dissymétrique avec un skew positif cependant
on observe une légère binomialité. Il y  a un tout petit pic autour de 55g de graisse saturée pour 100g de produit.

Les produits contenant de la graisse saturée vers ce pic et au-delà sont :
"""
filter_spe(nutri_data,'pnns_groups_1',(nutri_data['saturated-fat_100g']>55) & (nutri_data['saturated-fat_100g']<100))

"""
La plupart de ces produits sont du beurre, huile, sauce.
Il faut éviter de trop consommer ces produtis pour un régime équilibré.

Graisse
"""
hist_1(nutri_data,'fat_100g',xLabel='g de graisse pour 100g de produit')

"""

"""

#%%
# Les protéines
"""
2.1.2 - Les protéines
"""
hist_1(nutri_data,'proteins_100g',xLabel='g de protéines pour 100g de produit')
"""
Distribution dissimétrique avec un skew positif. Il y a peu de produits avec beaucoup de protéines, 
ce sont des compléments alimentaires pour la prise de masse musculaire :
"""
filter_spe(nutri_data,'product_name',(nutri_data['proteins_100g']>70) & (nutri_data['proteins_100g']<100))

#%%
# glucides
"""
2.1.3 - Les glucides 

Les glucides sont notamment composés du sucre
"""
hist_1(nutri_data,'sugars_100g',xLabel='g de sucre pour 100g de produit')
"""
Distribution dissimétrique avec un skew positif. Il y a peu de produits avec beaucoup de protéines, 
Les produits contenant beaucoup de sucre (>80g pour 100g) sont surtout des collations sucrées et des boissons sucrées
"""
filter_spe(nutri_data,'pnns_groups_1',(nutri_data['sugars_100g']>80) & (nutri_data['sugars_100g']<100))

"""
Aussi les hydrates de carbone

"""
hist_1(nutri_data,'carbohydrates_100g',xLabel='g d\'hydrate de carbone pour 100g de produit')
"""
Semble être bimodal
"""
filter_spe(nutri_data,'pnns_groups_1',(nutri_data['carbohydrates_100g']>59) & (nutri_data['carbohydrates_100g']<61))#collations/céréales
filter_spe(nutri_data,'pnns_groups_1',(nutri_data['carbohydrates_100g']<3))#sauce, boissons

#%%
#nutritional score

"""
2.2 - Score nutritionnelle

On s'intéresse maintenant au score nutritionnel. Il existe dans nos données un score nutri français et un score nutri uk.

"""
density_multi_1(nutri_data,['nutrition-score-fr_100g','nutrition-score-uk_100g'],xLabel = 'Score nutritionnel')

"""
Peu de différence entre les deux variables. On va conserver le score français par la suite.

Distribution bimodale avec un pic à 0 (cf. ci-dessous : boissons sans sucre, eau, pâtes, nouilles - neutre en terme de nutriment - 'bons nutriments' compensent les 'mauvais nutriments') et un pic entre 10 et 20.

"""
filter_spe(nutri_data,'product_name',(nutri_data['nutrition-score-fr_100g']>-0.5) & (nutri_data['nutrition-score-fr_100g']<0.5))

#%%
"""
2.3 - Autres variables

2.3.1 - Energie

On va s'intéresser aux autres variables qui sont inclus dans le score nutritionel (Energie (kJ) / sodium), le % de fruit, veg, nut n'est pas inclus (99% de NA)

Histogrammes (un en kJ un autre en kcal)
"""
nutri_data['kcal_100g'] = nutri_data['energy_100g'].apply(lambda x: x*0.239006)
new_data = nutri_data                 
hist_1(nutri_data,'energy_100g',xLabel='kJ pour 100g de produit',range=[0,4000])
hist_1(nutri_data,'kcal_100g',xLabel='kcal pour 100g de produit',range=[0,1000])


"""
3 pics + queue

Produits hauts en calories : beurre, huile, sauce
"""
filter_spe(nutri_data,'product_name',nutri_data['energy_100g']>4000)

#%%
"""
2.3.2 - Sodium

"""
hist_1(nutri_data,'sodium_100g',xLabel='g pour 100g de produit',range=[0,3])
"""
La plupart des produits contiennet peu de sodium (heureusement)
"""

#%%
"""
2.3.3 Marques
"""
#déterminer les 20 marques les plus représentées dans brands
most_freq_brands = mostFrequent(nutri_data,'brands',20)
hist_2(nutri_data,'brands',most_freq_brands)

"""
Plus fréquents : marques de granbdes surfaces

"""
#%%
"""
2.3.4 Groupes de produits (pnns_groups_1)
"""
#déterminer les 20 marques les plus représentées dans brands
most_freq_pnns1 = mostFrequent(nutri_data,'pnns_groups_1',11)
most_freq_pnns1.remove('unknown')
hist_2(nutri_data,'pnns_groups_1',most_freq_pnns1)

"""
Surtout des collations sucrées dans les données
"""
#%%---------------------- Bivariate Plots Section -------------------
"""
3. Bivariate Plots Section

Les corrélations entre variables :
"""
# Variables d'intérêts
lis_bivariates=['fat_100g','saturated-fat_100g','sugars_100g','proteins_100g','carbohydrates_100g','nutrition-score-fr_100g']
# plot tous les scatter plots de la liste au-dessus

super_scatter(nutri_data,lis_bivariates,3)
            
            

#%%
#Plot le heatmap de corrélation entre toutes les variables retenues
corr = nutri_data.corr(method='spearman',min_periods = 10000)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

"""
??? chelou corrélation pour les composantes des scores nutritionnelles okay à part protéïnes
"""

#%%
"""
3.1 - proteins and fat
"""
scatter(nutri_data,'proteins_100g','fat_100g')
#Compléments de protéines pour la muscu haut en protéines et faible en graisse
#Huile à l'autre extrême
filter_spe(nutri_data,'product_name',(nutri_data['fat_100g']>80) & (nutri_data['proteins_100g']<3))
print()
filter_spe(nutri_data,'product_name',(nutri_data['fat_100g']<3) & (nutri_data['proteins_100g']>80))


#%%
"""
3.2 Fat and unsaturated fat
"""
scatter(nutri_data,'saturated-fat_100g','fat_100g')

# la plupart des produits ayant de la graisse contient aussi de la graisse saturé. Certains produits contiennent énormément des deux.
filter_spe(nutri_data,'product_name',(nutri_data['fat_100g']>90) & (nutri_data['saturated-fat_100g']>90))


# surtout de l'huile de coco
#%%
"""
3.3 Carbohydrates et graisse
"""
scatter(nutri_data,'carbohydrates_100g','fat_100g')

"""
Notons l'absence de produits au milieu

"""
filter_spe(nutri_data,'product_name',(nutri_data['carbohydrates_100g']>99) & (nutri_data['fat_100g']<1)) # bonbons / sucre
print()
filter_spe(nutri_data,'product_name',(nutri_data['carbohydrates_100g']<1) & (nutri_data['fat_100g']>99)) # huile
#%%
"""
3.4 graisse saturées et score nutritionnel 
"""
scatter(nutri_data,'nutrition-score-fr_100g','saturated-fat_100g',xmin=-15,xmax=30)
# beaucoup de graisse saturé n'est pas considéré comme une alimentation saine --> reflété ici
#%%
"""
3.5 carbohydrates et score nutritionnel 
"""
scatter(nutri_data,'nutrition-score-fr_100g','carbohydrates_100g',xmin=-15,xmax=30)
# répartiition sur tout l'intervalle du score nutritionnel -> non indicatif si sain ou non
#%%
"""
3.6 proteins et score nutritionnel 
"""
scatter(nutri_data,'nutrition-score-fr_100g','proteins_100g',xmin=-15,xmax=30)
# répartiition sur tout l'intervalle du score nutritionnel -> non indicatif si sain ou non
#%%---------------------- Multivariate Plots Section -------------------
"""
Intéressons nous aux nutriments contenus dans chaque type de produit
"""
#on ne retient que les types de produits les plus fréquents
type_of_products = 'pnns_groups_1'
lis_prod = mostFrequent(nutri_data,type_of_products,9)#modifier pour enlever unknown
lis_prod.remove('unknown')
"""
4.1 densité type d'aliment fonction de qté sucre

"""


density_multi_2(nutri_data,'sugars_100g','pnns_groups_1',lis_prod) 

""""
Les collations sucrées contiennent beaucoup de sucre
"""
#%%
"""
4.2 densité type d'aliment fonction de qté protéines
"""

density_multi_2(nutri_data,'proteins_100g','pnns_groups_1',lis_prod) 

""""
Sans surprise le poisson, la viande et les oeufs + produits laitiers contiennent le plus de protéines
"""
#%%
"""
4.3 densité type d'aliment fonction de graisse
"""
density_multi_2(nutri_data,'fat_100g','pnns_groups_1',lis_prod) 

""""
collations sucrées et produits laitiers continennt le plus de graisse

"""
#%%
"""
4.4 densité type d'aliment fonction de qté glucide
"""
density_multi_2(nutri_data,'carbohydrates_100g','pnns_groups_1',lis_prod) 

"""
plus deglucides : collations sucrées

"""

#%%
"""
4.5 densité type d'aliment fonction de l'énergie
"""
density_multi_2(nutri_data,'kcal_100g','pnns_groups_1',lis_prod) 

"""
sauces / huiles le plus de calories

"""
#%%
"""
4.5 densité type d'aliment fonction du score nutritionnel
"""
density_multi_2(nutri_data,'nutrition-score-fr_100g','pnns_groups_1',lis_prod, xmin=-15,xmax=30) 

"""
Deux extrêmités : collations sucrées score élevé --> pas sain / fruits, légumes socre négatif --> sain
"""
#%%
"""
4.6 densité des marques en fonction du score nutri

"""
freq_brand=mostFrequent(nutri_data,'brands',5)
density_multi_2(nutri_data,'nutrition-score-fr_100g','brands',freq_brand, xmin=-15,xmax=30) 

"""
Ici toutes les marques sont des marques distributeurs == ils vendent le même type de produit, facile à comparer
On remarque que Casino proposent des produits relativement plus sains que les autres distributeurs
"""
#%%
"""
4.7 Distribution des scores nutritionel par type d'aliment et par marque

Pour 4 groupes d'aliments
"""
for prod in lis_prod:
    boxplot_multi(nutri_data[nutri_data['pnns_groups_1'].isin([prod])],'nutrition-score-fr_100g','brands',freq_brand,'Nutrition Score Fr_100g for '+prod)

#%%
"""
5 Final
"""
#super_boxplot(nutri_data,'pnns_groups_1',lis_prod,'brands',freq_brand,2)
boxplot_multi(nutri_data,'nutrition-score-fr_100g','brands',freq_brand)

"""
Pour acheter des produits sains ==> vaut mieux acheter les produits casino, U ==> on peut aussi regarder par type de produit pour savoir quel type d'aliment acheter
"""
#%%
