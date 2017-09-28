import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# On charge le dataset
house_data_raw = pd.read_csv('house_data.csv')
print (house_data_raw)
plt.plot(house_data_raw['price'], house_data_raw['surface'], 'ro', markersize=4)
plt.show()
"""
price = house_data_raw['price']
surface = house_data_raw['surface']

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(surface, price)
# regr.predict(donnee_test)
"""
