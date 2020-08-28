# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 22:09:04 2020

@author: Natan Steinbruch
"""

import numpy as np

#
#
#
#entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
#saidas = np.array([0,0,0,1])
        #operador AND
#entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
#saidas = np.array([0,1,1,1])
        #operador OR
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,1,0,1]).T
        #operador SE ENTAO

D = entradas.shape[1]
pesos = 2 * np.random.random(size=D)-1
bias = 2 * np.random.random()-1
LR = 0.1
errorTotal = 1
while errorTotal != 0:
    errorTotal = 0
    for x_n,y_n in zip(entradas,saidas):
        y_pred = x_n.dot(pesos) + bias
        y_pred = np.where(y_pred > 0, 1, 0)
        error = y_n - y_pred
        errorTotal += abs(error)
        pesos = pesos + LR*np.dot(error,x_n)
        bias = bias + LR * error
        print("Total de erros: " + str(errorTotal))

print("Pesos: ", pesos)
print("Bias: ",bias)
print('y_pred: {0}' .format(np.dot(entradas,np.array(pesos))+bias))

