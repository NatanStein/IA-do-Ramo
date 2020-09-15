# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:51:49 2020

@author: Natan Steinbruch
"""

import process_dataset as pdt
import funcao_ativacao as fa
import numpy as np
import cv2
#
#Bloco Principal
#
pesos1 = 2 * np.random.random((784, 53)) -1
pesos2 = 2 * np.random.random((53, 36)) -1
pesos3 = 2 * np.random.random((36, 25)) -1
pesos4 = 2 * np.random.random((25, 1)) -1

bias1 = np.zeros((1,53))
bias2 = np.zeros((1,36))
bias3 = np.zeros((1,25))
bias4 = np.zeros((1,1))


path = "C:\\Users\\natst\\OneDrive\\Natan Steinbruch\\IA-do-Ramo\\DataSet\\"
#Modifique o path para onde está a sua pasta DataSet

train,test,dataSet,train_saidas,test_saidas = pdt.process_data_set(path)

camada_oculta0 = fa.relu(np.dot(train,pesos1) + bias1)

camada_oculta1 = fa.relu(np.dot(camada_oculta0,pesos2) + bias2)

camada_oculta2 = fa.relu(np.dot(camada_oculta1,pesos3) + bias3)

camada_saida = fa.relu(np.dot(camada_oculta2,pesos4) + bias4)
camada_saida = np.where(camada_saida > 0,1,0)


