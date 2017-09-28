import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# On charge le dataset

def plot( house_data):
    plt.plot(house_data['price'], house_data['surface'], 'ro', markersize=4)
    plt.show()
    return;
def clean( house_data ):
    for index, row in house_data.iterrows():
        print row["price"], row["surface"], row["arrondissement"]
    return;

house_data_raw = pd.read_csv('house_data.csv')
clean(house_data_raw)
#plot (house_data_raw)

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
