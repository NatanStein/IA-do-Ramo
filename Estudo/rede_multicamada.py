# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:55:32 2020

@author: Natan Steinbruch
"""

import numpy as np

def sigmoide(soma):
    return 1 / (1+np.exp(-soma))

def sigmoideDerivada(sig):
    return sig * (1-sig)
    

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

saidas = np.array([[0],
                   [1],
                   [1],
                   [0]])

pesos0 = 2 * np.random.random((2,3)) - 1
pesos1 = 2 * np.random.random((3,1)) - 1

momento = 1
taxaAprendizagem = 0.3
epocas = 100000000
for j in range(epocas):   
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada,pesos0)
    camadaOculta = sigmoide(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta,pesos1)
    camadaSaida = sigmoide(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbs = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbs))
    if mediaAbs < 0.005:
        print(j)
        break
    derivadaSaida = sigmoideDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    derivadaOculta = sigmoideDerivada(camadaOculta)
    
    pesos1T = pesos1.T
    deltaSaidaXpesos1 = deltaSaida.dot(pesos1T)
    deltaOculta = derivadaOculta * deltaSaidaXpesos1
    
    entradaXdeltaSaida = camadaOculta.T.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (entradaXdeltaSaida * taxaAprendizagem)
    
    entradaXdeltaOculta = camadaEntrada.T.dot(deltaOculta)
    pesos0 = (pesos0 * momento) + (entradaXdeltaOculta * taxaAprendizagem)
    