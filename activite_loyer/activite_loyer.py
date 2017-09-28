import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def isNaN(num):
    return num != num
def plot( house_data):
    plt.plot(house_data['price'], house_data['surface'], 'ro', markersize=4)
    plt.show()
    return;
def print_( house_data ):
    for index, row in house_data.iterrows():
        print row["price"], row["surface"], row["arrondissement"]
    return;
def clean( house_data ):
    return house_data[house_data.price.notnull() & house_data.surface.notnull() & house_data.arrondissement.notnull()];
def clean_( house_data ):
    return house_data.dropna();

house_data_raw = pd.read_csv('house_data.csv')
house_data_raw_cleant = clean(house_data_raw)
house_data_raw_cleant_ = clean_(house_data_raw)
print(house_data_raw.shape)
print(house_data_raw_cleant.shape)
print(house_data_raw_cleant_.shape)

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
