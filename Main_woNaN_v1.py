# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 23:11:05 2017

@author: ATruong1
"""
save = False
#%%
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import get_dummies
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import decomposition
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#%%
def cleanAndSelect_v1(data):
    #selectionne les variables listées dans car_selected ['var1', 'var2', 'var3'], 
    var_selected =['language','plot_keywords','num_voted_users','actor_1_name','actor_2_name','actor_3_name','imdb_score','genres','duration','director_name','budget']
                                                            
    data_ = data[['movie_title','title_year']+var_selected]                                                       
    data_ = data_.dropna()
    data_ = data_.drop_duplicates(['movie_title','title_year'])
    data_.index.name = 'film_id'
    data_ = data_.reset_index()
    info = data_
    data_ = data_[var_selected].copy()

    df1 = info[['movie_title','film_id','genres','director_name','title_year']].head(10)
    df2 = info[['movie_title','film_id','genres','director_name','title_year']].sample(10)
    print(df1.append(df2))
    
    #Ajout des réalisateurs et des langues
    data_ = addColumnForEachContent(data_,'director_name', 5)
    data_ = addColumnForEachContent(data_,'language', 5)
    
    #Ajout des genres
    data_ = addColumnForEachWord(data_,'genres', 0)
      
    # Ajout des mots clés
    data_ = addColumnForEachWord(data_,'plot_keywords', 20)
      
    
    #Ajout des Acteurs (présence de l'acteur dans le film 1 ou 0, peu importe Acteur1, Acteur 2, Acteur3)
    data_['actors']= data_['actor_1_name']+'|'+data_['actor_2_name']+'|'+data_['actor_3_name']
    data_ = addColumnForEachWord(data_,'actors', 5)
    data_ = data_.drop(['actor_1_name','actor_3_name','actor_2_name'], axis=1)   
      
    #Nouveau score qui sublime les haut score avec beaucoup de votes et pénalise les score faibles avec beaucoup de vote
    score_ = (data['imdb_score']-data['imdb_score'].mean()).divide(10)
    num_voter_ = data['num_voted_users'].divide(data['num_voted_users'].max())
    data_['new_score'] = score_.multiply(num_voter_)
    data_ = data_.drop(['imdb_score'], axis = 1)

    return data_, info
#%%
def cleanAndSelect_v2(data):
    #selectionne les variables listées dans car_selected ['var1', 'var2', 'var3'], 
    var_selected =['num_user_for_reviews','num_voted_users','imdb_score','duration','budget','genres']
                                                            
    data_ = data[['movie_title','title_year','director_name']+var_selected]                                                       
    data_ = data_.dropna()
    data_ = data_.drop_duplicates(['movie_title','title_year'])
    data_.index.name = 'film_id'
    data_ = data_.reset_index()
    info = data_
    data_ = data_[var_selected].copy()

    df1 = info[['movie_title','film_id','genres','director_name','title_year']].head(10)
    df2 = info[['movie_title','film_id','genres','director_name','title_year']].sample(10)
    print(df1.append(df2))
    
    #Ajout des genres
    data_ = addColumnForEachWord(data_,'genres', 0)
     
       
    #Nouveau score qui sublime les haut score avec beaucoup de votes et pénalise les score faibles avec beaucoup de vote
    score_ = (data['imdb_score']-data['imdb_score'].mean()).divide(10)
    num_voter_ = data['num_voted_users'].divide(data['num_voted_users'].max())
    data_['new_score'] = score_.multiply(num_voter_)
    data_ = data_.drop(['imdb_score'], axis = 1)
         
    return data_, info
#%%
def cleanAndSelect_vf(data):
    #selectionne les variables listées dans car_selected ['var1', 'var2', 'var3'], 
    var_selected =['language','num_voted_users','actor_1_name','actor_2_name','actor_3_name','imdb_score','genres','duration','director_name']
                                                            
    data_ = data[['movie_title','title_year']+var_selected]                                                       
    data_ = data_.dropna()
    data_ = data_.drop_duplicates(['movie_title','title_year'])
    data_.index.name = 'film_id'
    data_ = data_.reset_index()
    info = data_
    data_ = data_[var_selected].copy()

    df1 = info[['movie_title','film_id','genres','director_name','title_year']].head(10)
    df2 = info[['movie_title','film_id','genres','director_name','title_year']].sample(10)
    print(df1.append(df2))
    
    #Ajout des réalisateurs et des langues
    data_ = addColumnForEachContent(data_,'director_name', 0)
    data_ = addColumnForEachContent(data_,'language', 0)
    
    #Ajout des genres
    data_ = addColumnForEachWord(data_,'genres', 0)
      
    # Ajout des mots clés
    #data_ = addColumnForEachWord(data_,'plot_keywords', 20)
      
    
    #Ajout des Acteurs (présence de l'acteur dans le film 1 ou 0, peu importe Acteur1, Acteur 2, Acteur3)
    data_['actors']= data_['actor_1_name']+'|'+data_['actor_2_name']+'|'+data_['actor_3_name']
    data_ = addColumnForEachWord(data_,'actors', 0)
    data_ = data_.drop(['actor_1_name','actor_3_name','actor_2_name'], axis=1)   
      
    #Nouveau score qui sublime les haut score avec beaucoup de votes et pénalise les score faibles avec beaucoup de vote
    score_ = (data['imdb_score']-data['imdb_score'].mean()).divide(10)
    num_voter_ = data['num_voted_users'].divide(data['num_voted_users'].max())
    data_['new_score'] = score_.multiply(num_voter_)
    data_ = data_.drop(['imdb_score'], axis = 1)
    
    return data_, info

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
def plotSilhouette(data, start = 200, end = 1000, step = 100, title_save=None):
    #plot silhouettes pour nb de centroids allant de 'start' à 'end'
    range_ = range(start, end, step)
    res =[]
    for k in range_:
        kmeans = KMeans(n_clusters=k).fit(data)
        res.append(metrics.silhouette_score(data,kmeans.labels_))
    plt.plot(range_,res,marker='o')
    plt.xlabel('Score de Silhouette')
    plt.savefig('scoreSilhouette_{}_{}_{}.png'.format(start,end,title_save))
    plt.show()
    
    return
def pca_plot(data, start = 2, end = 103, step = 10, title_save=None):
    lis=[]
    range_study = range(start,end,step)
    for k in range_study:
        pca = decomposition.PCA(n_components = k)
        pca.fit(data)
        lis.append(pca.explained_variance_ratio_.sum())
    plt.plot(range_study,lis, marker='o')
    plt.xlabel('Ratio de var expliquée vs nombre de dimensions')
    plt.savefig('PCA_{}_{}_{}.png'.format(start,end,title_save))
    plt.show()
    return
def distance_matrix(data):
    #Calcule une matrice de distance euclidienne entre toutes les lignes de 'data'
    dist_ = pdist(data, 'euclidean')
    dist_ = pd.DataFrame(squareform(dist_))
    return dist_
def getRecommendation_(index_bis , info, distanceMatrix):
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
    movie = info.iloc[[index_]]
    recommendations = getRecommendation_(index_,info,d_matrix)
    if movie is None or recommendations is None:
        print('Sorry, we are not able to recommend you a movie based on the selected movie')
    else:
        selected_columns_display = ['movie_title', 'genres','director_name','title_year']
        print_("Selected Movie:")
        print(movie[selected_columns_display].to_string(index=False,header=False))
        print_("Recommendations:")
        print(recommendations[selected_columns_display].to_string(index=False,header=False))
    return movie, recommendations
def recommend_clustering(data, info,film_id):
    index = info.index[info['film_id'] == film_id].tolist()
    if len(index) == 0:
        return None, None
    else:
        index_ = index[0]
        movie = info.iloc[[index_]]
        cluster = movie['cluster'].iloc[0]
        size_cluster = data[data['cluster']==cluster].shape[0]
        if size_cluster < 5:
            lis_index = data[data['cluster']==cluster].sort_values('new_score').head(size_cluster).index.tolist()
        else:
            if index_ in data[data['cluster']==cluster].sort_values('new_score').head(5).index.tolist():
                lis_index = data[data['cluster']==cluster].sort_values('new_score').head(6).index.tolist()
                lis_index.remove(index_)
            else:
                lis_index = data[data['cluster']==cluster].sort_values('new_score').head(5).index.tolist()
                
        recommendations = info.iloc[lis_index]
        
        if movie is None or recommendations is None:
            print('Sorry, we are not able to recommend you a movie based on the selected movie')
        else:
            selected_columns_display = ['movie_title', 'genres','director_name','title_year']
            print_("Selected Movie:")
            print(movie[selected_columns_display].to_string(index=False,header=False))
            print_("Recommendations:")
            print(recommendations[selected_columns_display].to_string(index=False,header=False))
            return movie, recommendations
def print_(string):
    # Format de print
    separator = "---------------------------"
    print(separator + " " + string + " " + separator)
    return
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
    return;
def scatter(data, columnName1, columnName2, xmin = 0, xmax = 100, ymin = 0, ymax = 100, save=False):
    # Pour analyse bi-dim, plot une variable contre une autre, affiche des petits points
    plt.scatter(data[columnName1], data[columnName2], color = '#CB6872', s = 10)
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
def linear_reg_param(data, lis_var, y_var):
    data_ = data[lis_var]
    ind_tot = data_.index.values.tolist()
    data_ = data_.dropna()
    ind = data_.index.values.tolist()
    ind_na = [x for x in ind_tot if x not in ind]
    y_ = data[y_var].iloc[ind]
    ind_y=y_.index.values.tolist()
    y_ = y_.dropna()
    index_na = [x for x in ind_y if x not in y_.index.values.tolist()]
    data_ = data_.drop(index_na)
    
    data_= normalize(data_) 
    
    
    # On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
    X = np.matrix(data_.as_matrix())
     
    y = np.matrix(y_).T
     
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)
     
    r_base = linear_model.LinearRegression()
    r_base.fit(xtrain, ytrain)
    y_predicted = r_base.predict(xtest)
     
       
    score_r2 = r2_score(ytest, y_predicted)
    MSE =mean_squared_error(ytest, y_predicted)
     
    print('Score r2: ',score_r2)
    print('Mean Squared Error: ',MSE)

    
    return score_r2, MSE, ind_na
def bar_chart(serie, title=None, ylabel=None, xlabel=None):
    serie.plot.bar()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    plt.savefig('bar_chart_{}.png'.format(title))
    return
def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data_normalized = pd.DataFrame(np_scaled)
    return data_normalized
def get_zeros(data):
    lis_zeros=[]
    data= data.drop(['color','director_name','actor_1_name','actor_2_name','actor_3_name','language','country','genres','movie_title','plot_keywords','movie_imdb_link','content_rating'],axis=1)
    for column in list(data):
        num_zero = data[column][data[column] ==0].count()
        lis_zeros.append(num_zero)
    data_ = pd.DataFrame(lis_zeros, index=data.columns.tolist())
    return data_.divide(data.shape[0]).multiply(100)
def pca_trans(data_norm, n_components):
    pca = decomposition.PCA(n_components)
    pca.fit(data_norm)
    data_trans = pca.transform(data_norm)
    pca.explained_variance_ratio_.sum()
    return data_trans
def print_var(data, info = None):
    #Affiche les variables dans le dataframe df, le data type, ainsi que les 5 premiers éléments si 'afficher'='oui'
    
    print('Variables')
    print()
    for var in list(data):
        print('-------------------------------------------------')
        #print(var,' (',df[var].dtype,')','     ',df[var][0],',',df[var][1],',',df[var][2],',',df[var][3],',',df[var][4]) # A améliorer
        print(var,' (',data[var].dtype,')','      ')
        if info == 'oui':
            print(data[var][0:4])
    print('-------------------------------------------------')
    print()
    print('Le nombre de variables est de :',len(list(data)))
    return
#%% -------------------------------- Chargement -----------------
pd.set_option('display.width', 1000)
data = pd.read_csv('movie_metadata.csv', sep=",")
#%% -------------------------------- Etude exploratoire -----------------
percent_na = data.isnull().sum().divide(data.shape[0]).sort_values(ascending = False).multiply(100)
bar_chart(percent_na, '% de valeurs inexistantes par variable', '%', 'Variables')
#%%
bar_chart(get_zeros(data), '% de valeurs nulles par variable','%','Variables')
#%%
data_dup = data.drop_duplicates()
nb_duplicates = data.shape[0]-data_dup.shape[0]  
    
#%% ------------------------- Prédisons les valeurs vides --------------
'''
Gross et Budget sont les deux variables avec le plus de NaN
Popularité d'un film ne peut être représentée par FB like parce que les vieux films n'ont pas forcément de de page fb
'''
heat_map(data,method='spearman',min_periods=100, save=True)
'''
Sans surprise le budget est très corrélé au Gross mais aussi du nombre de voted users
Le Gross est fortement corrélé au budget, num users for review, num voted users
'''
#%%
score_r2, MSE, ind_na = linear_reg_param(data[data['country']=='USA'].reset_index(), ['num_voted_users','budget'], 'gross')
'''
En prenant les variables les plus corrélées on obtient un score r2 d'environ 0,45 ce qui est très faible
On ne prendra pas le gross mais la varibale très corrélé à elle 'num_voted_users' cela fait sens : 
Plus de personnes ont vu le film, plus il y a de votes
'''
data[['num_voted_users','gross']].corr() #64%
#%%---------------------- Sélection de variables ----------------------
data_1, info_1 = cleanAndSelect_v1(data)

#%% rescaling des données et calcul dela matrice de distance
data_1_norm = normalize(data_1) 
#%% --------------------- Prédiction par clustering v1 (base complète filtrée) ----------------------
pca_plot(data_1_norm, 2, 253, 50,'v1') #78% pour 250 variables
#%%
data_1_trans = pca_trans(data_1_norm, 250)

#%%
plotSilhouette(data_1_trans, 2, 103, 25) #pas terrible, trop de variables categoriels et trop éparses
#%%------------------- Prédiction par clustering v2 (base tronquée) -------------------
#%%
data_2, info_2 = cleanAndSelect_v2(data)
data_2_norm = normalize(data_2) 
pca_plot(data_2_norm, 2, 29, 5,'v2') #86% pour 12 variables
#%%
data_2_trans = pca_trans(data_2_norm, 12)

#%%
plotSilhouette(data_2_trans, 2, 503, 100,'v2') #200 clusters, coef = 0,75
#%%
kmeans = KMeans(n_clusters=200).fit(data_2_trans)
#%%
labels = pd.Series(kmeans.labels_,index = data_2.index.tolist())
info_2 = pd.concat([info_2, labels.to_frame('cluster')], axis = 1)
data_2 = pd.concat([data_2, labels.to_frame('cluster')], axis = 1)
#%% 
# rajouter un histo des cluster avec moins de 5
film_id = 30

movie, recommendations = recommend_clustering(data_2, info_2, film_id)
#%% -------------------- Prédiction par matrice de distance avec base tronquée -------------------
dmatrix_2 = distance_matrix(data_2_norm)

#%% Recommandation avec la matrice de distance
'''id du film'''
# film_id = 9 Harry Potter
# film_id = 3 The Dark Knight Rises
# film_id = 2607 The King's Speech
# 283 Gladiator
film_id = 3

movie, recommendations = recommend(data_2, info_2, film_id, dmatrix_2)
#%% -------------------- Prédiction par matrice de distance avec base complète filtrée -------------------
dmatrix_1 = distance_matrix(data_1_norm)

#%% Recommandation avec la matrice de distance
'''id du film'''
# film_id = 9 Harry Potter
# film_id = 3 The Dark Knight Rises
# film_id = 2607 The King's Speech
# 283 Gladiator
film_id = 32

movie, recommendations = recommend(data_1, info_1, film_id, dmatrix_1)
#%% -------------------- Prédiction par matrice de distance avec base complète non filtrée (v_final) -------------------

data_f, info_f = cleanAndSelect_vf(data)
data_f_norm = normalize(data_f)                                                                                            
dmatrix_f = distance_matrix(data_f_norm)

#%% Recommandation avec la matrice de distance
'''id du film'''
# film_id = 9 Harry Potter
# film_id = 3 The Dark Knight Rises
# film_id = 2607 The King's Speech
# film_id = 283 Gladiator
# film_id = 32 Iron Man 3
# film_id = 30 Skyfall
# film_id = 26 Titanic
film_id = 26

movie, recommendations = recommend(data_f, info_f, film_id, dmatrix_f)