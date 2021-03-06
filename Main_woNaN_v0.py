# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 23:11:05 2017

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
from sklearn import decomposition
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
#%%
def cleanAndSelect(data, var_selected):
    #selectionne les variables listées dans car_selected ['var1', 'var2', 'var3'], 
    
    data = data[['movie_title','title_year']+var_selected]                                                       
    
    data = data.dropna()
    data = data.drop_duplicates(['movie_title','title_year'])
    data.index.name = 'film_id'
    data = data.reset_index()
    
    
    
    return data, data[var_selected].copy()
#%%
def feat_to_drop(data, threshold):
    # renvoie une liste des variables à garder après condition de fréquence avec 'threshold'
    filtered = data.sum(axis = 0) < threshold       
    return filtered.index[filtered].tolist()
def addColumnForEachWord(data,variable, threshold):
    #va créer une colonne pour chaque élément presént la colonne 'variable' (pour les listes séparées par '|')
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
def plotSilhouette(data, start = 200, end = 1000, step = 100):
    #plot silhouettes pour nb de centroids allant de 'start' à 'end'
    range_ = range(start, end, step)
    res =[]
    for k in range_:
        kmeans = KMeans(n_clusters=k).fit(data)
        res.append(metrics.silhouette_score(data_,kmeans.labels_))
    plt.plot(range_,res,marker='o')
    plt.xlabel('Score de Silhouette')
    plt.savefig('scoreSilhouette_{}_{}.png'.format(start,end))
    plt.show()
    return
def distance_matrix(data):
    #Calcule une matrice de distance euclidienne entre toutes les lignes de 'data'
    dist_ = pdist(data, 'euclidean')
    dist_ = pd.DataFrame(squareform(dist_))
    return dist_
def getRecommendation(index_bis , info, distanceMatrix):
    # renvoie les 5 éléments les plus proches de 'index_bis' dans 'distance_matrix'
    nsmallest_list = distanceMatrix.nsmallest(6, index_bis).index.values.tolist()
    del nsmallest_list[0]                                         
    res = info.iloc[nsmallest_list]
    return res
def recommend(data, info, film_id, d_matrix):
    #Renvoie les films recommandés par la fonction 'getRecommend'
    index = info.index[info['film_id'] == film_id].tolist()
    if len(index) == 0:
        return None, None
    else:
        index_ = index[0]
    return info.iloc[[index_]] , getRecommendation(index_,info,d_matrix)
def print_( string ):
    # Format de print
    separator = "---------------------------"
    print(separator + " " + string + " " + separator)
    return
#%% Chargement
pd.set_option('display.width', 1000)
data = pd.read_csv('movie_metadata.csv', sep=",")
#%% Clean et Observation
info, data_ = cleanAndSelect(data, ['num_voted_users','actor_1_name','actor_2_name','actor_3_name','imdb_score','genres','duration','gross','director_name','budget'])
df1 = info[['movie_title','film_id','genres','director_name','title_year']].head(10)
df2 = info[['movie_title','film_id','genres','director_name','title_year']].sample(10)
print(df1.append(df2))


#%% Ajout des genres
threshold = 0
data_ = addColumnForEachWord(data_,'genres', threshold)
#%% Ajout des mots clés
#data_ = addColumnForEachWord(data_,'plot_keywords', threshold)
#%% Ajout des réalisateurs
data_ = addColumnForEachContent(data_,'director_name', threshold)
#%% Ajout des Acteurs (présence de l'acteur dans le film 1 ou 0, peu importe Acteur1, Acteur 2, Acteur3)
data_['actors']= data_['actor_1_name']+'|'+data_['actor_2_name']+'|'+data_['actor_3_name']

data_ = addColumnForEachWord(data_,'actors', threshold)
data_ = data_.drop(['actor_1_name','actor_3_name','actor_2_name'], axis=1)   
#%% nouveau score qui sublime les haut score avec beaucoup de votes et pénalise les score faibles avec beaucoup de vote
score_ = data['imdb_score'].divide(10)
num_voter_ = (data['num_voted_users']-data['num_voted_users'].mean()).divide(data['num_voted_users'].max())
data_['new_score'] = score_.multiply(num_voter_)
data_ = data_.drop(['num_voted_users','imdb_score'], axis = 1)
#%% succès commmercial ou pas
data_['profitability'] = data_['gross'].divide(data_['budget'])
data_=data_.drop(data_['gross'])


#%% rescaling des données et calcul dela matrice de distance
data_scaled = preprocessing.scale(data_)

dmatrix = distance_matrix(data_scaled)

#%% Recommandation avec la matrice de distance
'''id du film'''
# film_id = 9 Harry Potter
# film_id = 3 The Dark Knight Rises
# film_id = 2607 The King's Speech
# 283 Gladiator
film_id = 2607

movie, recommendations = recommend(data_, info, film_id, dmatrix)
if movie is None or recommendations is None:
    print('Sorry, we are not able to recommend you a movie based on the selected movie')
else:
    selected_columns_display = ['movie_title', 'genres','director_name','title_year']
    print_("Selected Movie:")
    print(movie[selected_columns_display].to_string(index=False,header=False))
    print_("Recommendations:")
    print(recommendations[selected_columns_display].to_string(index=False,header=False))
#%%
#%%
lis=[]
for k in range(2,1004,250):
    pca = decomposition.PCA(n_components = k)
    pca.fit(data_scaled)
    lis.append(pca.explained_variance_ratio_.sum())
plt.plot(range(2,1004,250),lis, marker='o')
plt.xlabel('Ratio de var expliquée vs nombre de dimensions')


#%%
pca = decomposition.PCA(n_components = 1500) #70% expliqué
pca.fit(data_scaled)
data_trans = pca.transform(data_scaled)
pca.explained_variance_ratio_.sum()

#%%
plotSilhouette(data_trans, 2, 1003, 200)
