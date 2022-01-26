import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.neighbors import KNeighborsClassifier

# ucitavanje skupa i prvih/poslednjih 5 redova
data = pd.read_csv('car_state.csv')

print(data.head())
print(data.tail())

# koncizne informacije i statisticke informacije
print(data.info())
print(data.describe(include=[object]))

# graficki prikaz zavisnosti
fig, ax = plt.subplots(3, 2, sharey=True)
fig.suptitle('Graficki prikaz zavisnosti')
output = 'status'
data.plot(x='maintenance', y=output,
          kind='scatter', ax=ax[0, 0])
data.plot(x='doors', y=output,
          kind='scatter', ax=ax[1, 0])
data.plot(x='seats', y=output,
          kind='scatter', ax=ax[2, 0])
data.plot(x='trunk_size', y=output,
          kind='scatter', ax=ax[0, 1])
data.plot(x='safety', y=output,
          kind='scatter', ax=ax[1, 1])
data.plot(x='buying_price', y=output,
          kind='scatter', ax=ax[2, 1])
plt.show()
# odabir atributa
# not usefull:
# usefull: buying_price, maintenance, doors, seats, trunk_size, safety
data_labels = ['buying_price', 'maintenance', 'doors', 'seats',
               'trunk_size', 'safety']

data_train = data[data_labels].copy()
y = data[output]

# transformacije
trunk_size_map = {'small': 1, 'medium': 2, 'big': 3}
high_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4}

le = LabelEncoder()
data_train['doors'] = le.fit_transform(data_train.doors)
data_train['seats'] = le.fit_transform(data_train.seats)
data_train['trunk_size'] = data_train['trunk_size'].map(trunk_size_map)
data_train['safety'] = data_train['safety'].map(high_map)
data_train['maintenance'] = data_train['maintenance'].map(high_map)
data_train['buying_price'] = data_train['buying_price'].map(high_map)

data_train = (data_train-data_train.mean())/data_train.std()

# Treniranje
x_train, x_test, y_train, y_test = train_test_split(
    data_train, y, train_size=0.7)

# knn custom
k = math.floor(len(x_train.index)**(1/2))
if (k % 2 != 1):
    k += 1


def distance(x: pd.DataFrame, i1, x2: pd.DataFrame, i2):
    s = 0
    for j in data_labels:
        s += (x[j][i1] - x2[j][i2])**2
    return (s)**(1/2)


def predict(x: pd.DataFrame, id):
    pass
    vals = []
    for i in x_train.index:
        vals.append([distance(x, id, x_train, i), y_train[i]])

    vals.sort(key=lambda val: val[0])

    res = {}
    for i in range(k):
        try:
            res[vals[i][1]] += 1
        except KeyError:
            res[vals[i][1]] = 1

    max = 0
    result = ''
    for key in res.keys():
        if(res[key] > max):
            max = res[key]
            result = key
    return result


# knn built-in
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

# poredjenje
statuses = {
    'unacceptable': 0,
    'acceptable': 1,
    'good': 2,
    'very good': 3
}


def loss(y_pred, y: pd.DataFrame):
    s = 0
    for i, index in enumerate(y.index):
        s += abs(statuses[y[index]] - statuses[y_pred[i]])
    return s/len(y.index)


def accuracy(y_pred, y: pd.DataFrame):
    s = 0
    for i, index in enumerate(y.index):
        if (y[index] == y_pred[i]):
            s += 1
    return s/len(y.index)*100


y_pred_custom = [predict(x_test, i) for i in x_test.index]
y_pred_sklearn = neigh.predict(x_test)

print('My      | Loss: ', loss(y_pred_custom, y_test))
print('Sklearn | Loss: ', loss(y_pred_sklearn, y_test))
print('My      | Accuracy: ', accuracy(y_pred_custom, y_test), '%')
print('Sklearn | Accuracy: ', accuracy(y_pred_sklearn, y_test), '%')
