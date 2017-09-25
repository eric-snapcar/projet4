# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# On charge le dataset
house_data_raw = pd.read_csv('house.csv')
house_data = house_data_raw[house_data_raw['loyer']<7000]
# On affiche le nuage de points dont on dispose
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
#plt.show()
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].as_matrix()]).T
y = np.matrix(house_data['loyer']).T
print(np.ones(house_data.shape[0]))
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(theta)
plt.xlabel('Surface')
plt.ylabel('Loyer')

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)

# On affiche la droite entre 0 et 250
plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')

#plt.show()


print(theta.item(0) + theta.item(1) * 35)
