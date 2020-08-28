# -*- coding: utf-8 -*-
"""
Created on Mon May  4 02:45:09 2020

@author: Natan Steinbruch
"""


import numpy as np
from sklearn import datasets

def sigmoide(soma):
    return 1 / (1+np.exp(-soma))

def sigmoideDerivada(sig):
    return sig * (1-sig)

base = datasets.load_breast_cancer()
    

entradas = base.data
valoresSaidas = base.target
saidas = np.empty([569,1], dtype=int)
for i in range(569):
    saidas[i] = valoresSaidas[i]

#entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
#saidas = np.array([[0],[1],[1],[0]])

pesos0 = 2 * np.random.random((30,15)) - 1
pesos1 = 2 * np.random.random((15,15)) - 1
pesos2 = 2 * np.random.random((15,1)) - 1

momento = 1
taxaAprendizagem = 0.4
epocas = 100000
for j in range(epocas):   
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada,pesos0)
    camadaOculta0 = sigmoide(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta0,pesos1)
    camadaOculta1 = sigmoide(somaSinapse1)
    
    somaSinapse2 = np.dot(camadaOculta1,pesos2)
    camadaSaida = sigmoide(somaSinapse2)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbs = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbs))
    if mediaAbs < 0.005:
        print(j)
        break
    derivadaSaida = sigmoideDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    derivadaOculta1 = sigmoideDerivada(camadaOculta1)
    pesos2T = pesos2.T
    deltaSaidaXpesos1 = deltaSaida.dot(pesos2T)
    deltaOculta1 = derivadaOculta1 * deltaSaidaXpesos1
    
    derivadaOculta0 = sigmoideDerivada(camadaOculta0)
    pesos1T = pesos1.T
    deltaOculta1Xpesos1 = deltaOculta1.dot(pesos1T)
    deltaOculta0 = derivadaOculta0 * deltaOculta1Xpesos1
    
    entradaXdeltaSaida = camadaOculta1.T.dot(deltaSaida)
    pesos2 = (pesos2 * momento) + (entradaXdeltaSaida * taxaAprendizagem)
    
    entradaXdeltaOculta1 = camadaOculta0.T.dot(deltaOculta1)
    pesos1 = (pesos1 * momento) + (entradaXdeltaOculta1 * taxaAprendizagem)
    
    entradaXdeltaOculta0 = camadaEntrada.T.dot(deltaOculta0)
    pesos0 = (pesos0 * momento) + (entradaXdeltaOculta0 * taxaAprendizagem)