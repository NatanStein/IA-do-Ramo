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
peso1 = 2 * np.random.random((784, 53)) -1
peso2 = 2 * np.random.random((53, 36)) -1
peso3 = 2 * np.random.random((36, 25)) -1
peso4 = 2 * np.random.random((25, 1)) -1

bias1 = np.zeros((1,53))
bias2 = np.zeros((1,36))
bias3 = np.zeros((1,25))
bias4 = np.zeros((1,1))


path = "C:\\Users\\natst\\OneDrive\\Natan Steinbruch\\IA-do-Ramo\\DataSet\\"
#Modifique o path para onde está a sua pasta DataSet

train,test,dataSet,train_saidas,test_saidas = pdt.process_data_set(path)
img = cv2.imread(path+"train/without_mask/83.jpg",0)
img = cv2.resize(img,(28,28)) / 255
img = img.flatten().reshape((1,784)) #Transforma a matriz 28x28 para um vetor linear
