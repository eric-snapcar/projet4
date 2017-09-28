import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

house_data_raw = pd.read_csv('house_data.csv')
house_data_raw_clean = clean(house_data_raw)
house_data_1 = house_data_raw_clean[['price','surface']]
house_data_1_train, house_data_1_test = train_test_split(house_data_1, train_size=0.8)
print(house_data_1_train)
print(house_data_1_test)




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
