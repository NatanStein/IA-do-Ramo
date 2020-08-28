# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:00:49 2020

@author: Natan Steinbruch
"""

import numpy as np

def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    #dot = produto escalar
    return stepFunction(s)

def treinar():
    erroTotal = 1
    while erroTotal != 0:
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.array(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print("Peso atualizado: " + str(pesos[j]))
        print("Total de erros: " + str(erroTotal))
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
saidas = np.array([0,1,0,1])
        #operador SE ENTAO
pesos = np.array([0.0,0.0])
taxaAprendizagem = 0.1

treinar()

print(calculaSaida(np.array(entradas[0])))
print(calculaSaida(np.array(entradas[1])))
print(calculaSaida(np.array(entradas[2])))
print(calculaSaida(np.array(entradas[3])))




