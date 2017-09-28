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
def getArrondissement( house_data_raw ):
    return house_data_raw.arrondissement.unique();
def linearRegression_1( house_data , plot):
    # Linear Regresssion No Arrondissement
    # Cleaning
    house_data = clean(house_data)
    # Column Selection
    house_data = house_data[['price','surface']]
    # Train Test
    house_data_train, house_data_test = train_test_split(house_data, train_size=0.8)
    # Train Test Surface Price
    house_data_train_surface = house_data_train.surface.values.reshape(-1,1)
    house_data_train_price = house_data_train.price.values.reshape(-1,1)
    house_data_test_surface = house_data_test.surface.values.reshape(-1,1)
    house_data_test_price = house_data_test.price.values.reshape(-1,1)
    rl = linear_model.LinearRegression()
    rl.fit(house_data_train_surface, house_data_train_price)
    house_data_predicted_price = rl.predict(house_data_test_surface)
    # Prediciton Error
    mean_squared_error_ = mean_squared_error(house_data_test_price, house_data_predicted_price)
    variance_score = r2_score(house_data_test_price, house_data_predicted_price)
    coefficient = rl.coef_
    theta0 = rl.predict([[0]])[0,0]
    print("Mean squared error: %.2f"  % mean_squared_error_)
    print('Variance score: %.2f' % variance_score)
    print('Coefficients: %.2f' % coefficient)
    print('Theta_0: %.2f' % theta0)
    # Plot
    if plot:
        plt.plot( house_data_train_surface, house_data_train_price,'ro', markersize=4)
        plt.plot( house_data_test_surface, house_data_predicted_price,'b', markersize=4)
        plt.show()
    return ;

house_data_raw = pd.read_csv('house_data.csv')
house_data_raw = house_data_raw[house_data_raw.price<1000]
linearRegression_1(house_data_raw,False)


house_data_raw_arrondissment = getArrondissement(clean(house_data_raw))
print(house_data_raw_arrondissment)



"""
plt.plot(house_data_raw['price'], house_data_raw['surface'], 'ro', markersize=4)
plt.show()
"""
"""
price = house_data_raw['price']
surface = house_data_raw['surface']

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(surface, price)
# regr.predict(donnee_test)
"""
