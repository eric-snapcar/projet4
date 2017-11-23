# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:29:25 2017

@author: ATruong1
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#%%
def concat():
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    lis_var= ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'ARR_DELAY', 'DISTANCE', 'CRS_ELAPSED_TIME']
    var_ref =['ORIGIN_AIRPORT_ID','ORIGIN','ORIGIN_CITY_NAME']
    data_=[]
    data_ref=[]
    for i in months:
        print('début '+i)
        data_int = pd.read_csv('2016_'+i+'.csv', sep=",",error_bad_lines=False)
        data_int_ = data_int[data_int.columns.intersection(lis_var)]
        ref_int = data_int[data_int.columns.intersection(var_ref)].drop_duplicates()
        data_.append(data_int_)
        data_ref.append(ref_int)
        print('fin '+i)
    data_ = pd.concat(data_)
    data_ref = pd.concat(data_ref)
    data_ref = data_ref.drop_duplicates()
    print('Il y a ',data_['ARR_DELAY'].isnull().sum(),' lignes où \'ARR_DELAY\' a des valeurs vides pour ',data_.shape[0],' lignes')
    data_ = data_[np.isfinite(data_['ARR_DELAY'])]
    print(data_.describe())
    data_ = data_.dropna() # que 2 lignes sur + de 5m où on a des NA
    return data_, data_ref
def print_var(products, info = None):
    #Affiche les variables dans le dataframe df, le data type, ainsi que les 5 premiers éléments si 'afficher'='oui'
    
    print('Variables')
    print()
    for var in list(products):
        print('-------------------------------------------------')
        print(var,' (',products[var].dtype,')','      ')
        if info == 'oui':
            print(products[var][0:4])
    print('-------------------------------------------------')
    print()
    print('Le nombre de variables est de :',len(list(products)))
    return
def bar_chart(serie, title=None, ylabel=None, xlabel=None):
    serie.plot.bar()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    plt.savefig('bar_chart_{}.png'.format(title))
def heat_map(data, method, min_periods, save=False):
    corr = data.corr(method,min_periods)
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
        
    if save:
        plt.tight_layout()
        plt.title('Matrice de corrélation des variables')
        plt.savefig('heatmap.png')
        plt.close()
    else:
        plt.title('Matrice de corrélation des variables')
        plt.show()
    return

#%% ------------------------------- Etude exploratrice sur la base du mois de janvier ------------------------------------
data_1 = pd.read_csv('2016_01.csv', sep=",",error_bad_lines=False)
#%%
percent_na = data_1.isnull().sum().divide(data_1.shape[0]).sort_values(ascending = False).multiply(100)
print(percent_na)
list_na_to_drop = percent_na[percent_na > 5 ].index.tolist()
data_1_ = data_1.drop(list_na_to_drop,axis=1)
heat_map(data_1,method='spearman',min_periods=100)
#%% On va sélectionner les variables restantes à la main
# On dégage toutes les variables relatives à la performance de l'arrivée et du départ et les raisons du retard.
# nous allons conserver les variables suivantes
lis_var= ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'ARR_DELAY', 'DISTANCE', 'CRS_ELAPSED_TIME']
# on va aussi garder un tableau de correspondance avec le nom de l'aéroport de départ et d'arrivée 

data_test=data_1_[data_1_.columns.intersection(lis_var)]
#%% ---------------------------------- Utilisons la vraie base ----------------------
data_, data_ref=  concat()
