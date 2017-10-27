# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:15:30 2017

@author: ATruong1
"""
#%%
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from pandas import get_dummies
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
#%%
def cleanAndSelect(data, var_selected, thresh1, thresh2):
    #selectionne les variables listées dans car_selected ['var1', 'var2', 'var3'], 
    # on enlève les films qui ont un % 'thresh1' de na en ligne et un % 'thresh2' de na en colonne
    selected_columns = var_selected
    data = data[selected_columns]
    data = data.dropna(axis=0,thresh=data.shape[1]*thresh1)
    data = data.dropna(axis=1,thresh=data.shape[1]*thresh2)
    return data
#%%
def feat_to_drop(data, threshold):
    filtered = data.sum(axis = 0) < threshold       
    return filtered.index[filtered].tolist()
def addColumnForEachWord(data,variable, threshold):
    #va créer une colonne pour chaque élément presént la colonne 'variable'
    def getWords(data):
        # obtiends une liste de tous les genres présents dans la colonne 'Genres' (unique)
        serie = data[variable].str.split('|').dropna()
        serie_ = serie.agg(['sum'])
        return list(set(serie_.values[0]))
    def add_column_eachWord(data, words):
        # rajoute une colonne pour chaque genre de film
        for word in words:
            data[word] = 0
        return data
    def add_column_wordsSplit(data):
        word_split = data[data[variable].isnull()==False][variable].apply(split_)
        data['words_split'] = word_split
        return data
    def fill_column_eachWord(row):
        #remplis les colonnes avec des 1 et des 0 si le genre correspond au film
        if pd.isnull(row[variable]) == False:
            words = row.words_split
            for word in words:
                row[word] = 1
        return row
    def split(string , separator):
        # parse un string avec des séparateurs
        return string.split(separator)
    def split_(string):
        return split(string,'|')
       
    data_ = add_column_eachWord(data,getWords(data))
    data_ = add_column_wordsSplit(data)
    data_ = data_.apply(fill_column_eachWord, axis = 1)
    data_ = data_.drop(['words_split',variable], axis=1)
    data_ = data_.drop(feat_to_drop(data_[getWords(data)], threshold), axis = 1)
    return data_
def addColumnForEachContent(data,variable, threshold):
    # renvoie une colonne pour chque élément dans la colonne 'variable'
    data_sup = get_dummies(data[variable])
    data_ = pd.concat([data, data_sup], axis=1)
    data_ = data_.drop(variable, axis=1)
    data_ = data_.drop(feat_to_drop(data_sup, threshold), axis=1)
    return data_
def plotSilhouette(data,x_min=2,x_max=10,x_step=1):
    # plot le K-means de 2 à 20 clusters et renvoie le coefficient de sil'houette pour chaque
    range_ = range(x_min,x_max,x_step)
    res =[]
    for k in range_:
        kmeans = KMeans(n_clusters=k).fit(data)
        res.append(metrics.silhouette_score(data_,kmeans.labels_))
    plt.plot(range_,res,marker='o')
    plt.show()
    return
#%%
data = pd.read_csv('movie_metadata.csv', sep=",")
#%%
data_ = cleanAndSelect(data, ['actor_1_name','actor_3_name','actor_2_name','plot_keywords','imdb_score','genres','duration','gross','director_name','budget','title_year'], 0.75, 0.75)
#%%
threshold = 5
data_ = addColumnForEachWord(data_,'genres', threshold)
#%%
data_ = addColumnForEachWord(data_,'plot_keywords', threshold)
#%%
data_ = addColumnForEachContent(data_,'director_name', 3)
#%%
data_['actors']= data_['actor_1_name']+'|'+data_['actor_2_name']+'|'+data_['actor_3_name']

data_ = addColumnForEachWord(data_,'actors', 3)
#%%
data_ = data_.drop(['actor_1_name','actor_3_name','actor_2_name'], axis=1)   
#%%          
data_1 = preprocessing.imputer(data_)
#%%
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data_)
data1=imp.transform(data_)
#%%
data_scaled = preprocessing.scale(data1)
#%%
plotSilhouette(data_scaled,x_min=2,x_max=502,x_step=100)