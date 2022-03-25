#!/usr/bin/python
# coding=utf-8

import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

datasets_X = []
datasets_Y = []
fr = open('../../dataset/regression/house_prices.txt', 'r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))

length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length, 1])
datasets_Y = np.array(datasets_Y)

minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX, maxX).reshape([-1, 1])

linear = linear_model.LinearRegression()
linear.fit(datasets_X, datasets_Y)

print('Data Length:', length)
print('Coefficients：', linear.coef_)
print('Intercept：', linear.intercept_)

plt.scatter(datasets_X, datasets_Y, color='green')
plt.plot(X, linear.predict(X), color='red')
plt.xlabel('House Area')
plt.ylabel('House Price')
plt.show()

