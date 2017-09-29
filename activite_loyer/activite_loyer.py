import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot( house_data):
    plt.plot(house_data['price'], house_data['surface'], 'ro', markersize=4)
    plt.show()
    return;
def print_( house_data ):
    for index, row in house_data.iterrows():
        print row["price"], row["surface"], row["arrondissement"]
    return;
def clean_old( house_data ):
    return house_data[house_data.price.notnull() & house_data.surface.notnull() & house_data.arrondissement.notnull()];
def clean( house_data ):
    return house_data.dropna();
def getTrainTestData( house_data ):
    house_data_train, house_data_test = train_test_split(house_data, train_size=0.8)
    house_data_train_surface = house_data_train.surface.values.reshape(-1,1)
    house_data_train_price = house_data_train.price.values.reshape(-1,1)
    house_data_test_surface = house_data_test.surface.values.reshape(-1,1)
    house_data_test_price = house_data_test.price.values.reshape(-1,1)
    return house_data_train_surface,house_data_train_price,house_data_test_surface,house_data_test_price;
def getArrondissement( house_data_raw ):
    return house_data_raw.arrondissement.unique();
def linearRegression_1( house_data , plot,graphTitle):
    # Cleaning
    house_data = clean(house_data)
    # Column Selection
    house_data = house_data[['price','surface']]
    # Train Test
    house_data_train_surface,house_data_train_price,house_data_test_surface,house_data_test_price = getTrainTestData(house_data)
    # Linear Regression
    linear_regresesion = linear_model.LinearRegression()
    linear_regresesion.fit(house_data_train_surface, house_data_train_price)
    house_data_predicted_price = linear_regresesion.predict(house_data_test_surface)
    # Prediciton Error
    mean_squared_error_ = mean_squared_error(house_data_test_price, house_data_predicted_price)
    variance_score = r2_score(house_data_test_price, house_data_predicted_price)
    coefficient = linear_regresesion.coef_
    theta0 = linear_regresesion.predict([[0]])[0,0]
    # Print Results
    print("Erreur: %.2f"  % mean_squared_error_)
    print('Variance: %.2f' % variance_score)
    print('Theta_0: %.2f' % theta0)
    print('Theta_1: %.2f' % coefficient)
    # Plot
    if plot:
        plt.title(graphTitle)
        plt.plot( house_data_train_surface, house_data_train_price,'ro', markersize=4)
        plt.plot( house_data_test_surface, house_data_predicted_price,'b', markersize=4)
        plt.show()
    return ;

def linearRegression_2( house_data_clean , plot):
    arrondissements = getArrondissement(house_data_clean)
    for arrondissement in arrondissements:
        house_data_raw_arrondissment = house_data_clean[(house_data_clean.arrondissement == arrondissement)]
        linearRegression_2_arrondissement(arrondissement,house_data_raw_arrondissment,plot)
    return;

def linearRegression_2_arrondissement( arrondissement, house_data, plot):
    print('--------------')
    print('Arrondissement: %s' % arrondissement)
    #print(house_data)
    linearRegression_1(house_data,plot,'Arrondissement: %s' % arrondissement)
    return;

house_data_raw = pd.read_csv('house_data.csv')
house_data_raw = house_data_raw[house_data_raw.price<7000]
house_data_clean = clean(house_data_raw)
print('--------------')
print('Tous les arrondissements')
linearRegression_1(house_data_raw,True,'Tous les arrondissements')
linearRegression_2(house_data_clean,True)
