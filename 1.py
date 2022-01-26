import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# ucitavanje skupa i prvih/poslednjih 5 redova
data = pd.read_csv('car_purchase.csv')

print(data.head())
print(data.tail())

# koncizne informacije i statisticke informacije
print(data.info())
print(data.describe(include=[object]))

# graficki prikaz zavisnosti
fig, ax = plt.subplots(3, 2, sharey=True)
fig.suptitle('Graficki prikaz zavisnosti')
data.plot(x='gender', y='max_purchase_amount',
          kind='scatter', ax=ax[0, 0])
data.plot(x='age', y='max_purchase_amount',
          kind='scatter', ax=ax[1, 0])
data.plot(x='annual_salary', y='max_purchase_amount',
          kind='scatter', ax=ax[2, 0])
data.plot(x='credit_card_debt', y='max_purchase_amount',
          kind='scatter', ax=ax[0, 1])
data.plot(x='net_worth', y='max_purchase_amount',
          kind='scatter', ax=ax[1, 1])
data.plot(x='customer_id', y='max_purchase_amount',
          kind='scatter', ax=ax[2, 1])
plt.show()

# odabir atributa
# not usefull: customer_id
# usefull: gender, age, annual_salary, credit_card_debt, net_worth
data_labels = ['gender', 'age', 'annual_salary',
               'credit_card_debt', 'net_worth']

data_train = data[data_labels].copy()
y = data['max_purchase_amount']

# transformacije
le = LabelEncoder()
data_train['gender'] = le.fit_transform(data_train.gender)
data_train = (data_train-data_train.mean())/data_train.std()


# Treniranje
x_train, x_test, y_train, y_test = train_test_split(
    data_train, y, train_size=0.7)

# gradijentni spust custom
w = np.array([0, 0, 0, 0, 0, 0])


def h(x: pd.DataFrame, k):
    h = w[0]
    for i, label in enumerate(data_labels):
        h += w[i + 1]*x[label][k]
    return h


def grad(x: pd.DataFrame, y: pd.DataFrame):
    n = len(x.index)
    dw = []
    # dw0
    dw0 = w[0] + (sum([w[j] * x[data_labels[j-1]].sum()
                       for j in range(1, 6)]) - y.sum())/n
    dw.append(dw0)
    # dw 1..n
    for i in range(1, 6):
        dw.append(
            (

                - sum(
                    [y[k]*x[data_labels[i-1]][k]
                     for k in x.index]
                )
                + x[data_labels[i-1]].sum() * w[0]
                + sum(
                    [w[j]*sum(
                        [x[data_labels[j-1]][k]*x[data_labels[i-1]][k]
                         for k in x.index]
                    )for j in range(1, 6)]
                )
            ) / n
        )
    return np.array(dw)


maxiter = 1e6
a = 1
min_diff = 1e-9

for i in range(int(maxiter)):
    dw = grad(x_train, y_train)
    w = w - a*dw
    if not [x for x in dw if x > min_diff]:
        break


# gradijentni spust built-in
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# poredjenje


def loss(h, y):
    return ((y - h)**2)/2


def score_custom(x:  pd.DataFrame, y_true:  pd.DataFrame):
    y_pred = np.array([h(x, i) for i in x.index])

    return 1 - ((y_true - y_pred) ** 2).sum()/((y_true - y_true.mean()) ** 2).sum()


def test_custom(x: pd.DataFrame, y:  pd.DataFrame):
    return sum([loss(h(x, i), y[i]) for i in x.index])/len(x.index)


def test_built_in(x: pd.DataFrame, y: pd.DataFrame):
    h = regressor.predict(x)
    return sum([loss(h[i], y[ind]) for i, ind in enumerate(x.index)])/len(x.index)


print('SKLearn| score[train]: ', regressor.score(x_train, y_train))
print('SKLearn| score[test] : ', regressor.score(x_test, y_test))
print('My     | score[train]: ', score_custom(x_train, y_train))
print('My     | score[test] : ', score_custom(x_test, y_test))

print('My coefficients     : ', w.tolist())
print('SKLearn coefficients: ', regressor.coef_.tolist())


print("My      | Test Data Loss:  ", test_custom(x_test, y_test))
print("My      | Train Data Loss: ", test_custom(x_train, y_train))

print("SKLearn | Test Data Loss:  ", test_built_in(x_test, y_test))
print("SKLearn | Train Data Loss: ", test_built_in(x_train, y_train))
