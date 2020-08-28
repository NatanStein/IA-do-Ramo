# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 01:35:14 2020

@author: Natan Steinbruch
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets.samples_generator import make_blobs

df = pd.read_csv("notas.csv")

entradas = df[['prova1', 'prova2', 'prova3']].values
saidas = df['final'].values.reshape(-1, 1)
minmax = MinMaxScaler(feature_range=(-1,1))
entradas = minmax.fit_transform(entradas.astype(np.float64))

D = entradas.shape[1]
pesos = [2*random() - 1 for i in range(D)]
bias = 2 * np.random.random()-1

LR = 1e-2

for step in range(2001):
    cost = 0
    for x_n, y_n in zip(entradas, saidas):
        y_pred = sum([x_i*w_i for x_i, w_i in zip(x_n, pesos)]) + bias
        error = y_n - y_pred
        pesos = [w_i + LR*error*x_i for x_i, w_i in zip(x_n, pesos)]
        bias = bias + LR*error
        cost += error**2
        print('step {0}: {1}'.format(step, cost))

print('w: ', pesos)
print('b: ', bias)