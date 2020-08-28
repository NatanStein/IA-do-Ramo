# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 02:21:48 2020

@author: Natan Steinbruch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

def sigmoid (x):
    return 1.0/ (1.0 + np.exp(-x))

#
#Bloco Principal
#
    
df = pd.read_csv('anuncios.csv')
entradas, saidas = df[['idade','salario']].values, df.comprou.values.reshape(-1,1)
minmax = MinMaxScaler(feature_range=(-1,1))
entradas = minmax.fit_transform(entradas.astype(np.float64))



D = entradas.shape[1]
pesos = 2*np.random.random((1, D))-1
bias = 2 * np.random.random()-1
LR = 0.01

for steps in range(1000):
    z = np.dot(entradas,pesos.T) + bias
    y_pred = sigmoid(z)
    error = saidas - y_pred
    
    pesos = pesos + LR*np.dot(error.T,entradas)
    bias = bias + LR*sum(error)
    cost = np.mean(-saidas*np.log(y_pred) - (1-saidas)*np.log(1-y_pred))
    print('Step {0}: {1}' .format(steps,cost))

print('Pesos: {0}' .format(pesos))
print('Bias: {0}' .format(bias))

x = entradas
y = saidas
w = pesos
b = bias
x1 = np.linspace(x[:, 0].min(), x[:, 0].max())
x2 = np.linspace(x[:, 1].min(), x[:, 1].max())
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
x1_mesh = x1_mesh.reshape(-1, 1)
x2_mesh = x2_mesh.reshape(-1, 1)

x_mesh = np.hstack((x1_mesh, x2_mesh))
y_pred = sigmoid(np.dot(x_mesh, w.T) + b)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x[:,0], x[:,1], y, c=y.ravel())
ax.plot_trisurf(x1_mesh.ravel(), x2_mesh.ravel(), y_pred.ravel(), alpha=0.3, shade=False)