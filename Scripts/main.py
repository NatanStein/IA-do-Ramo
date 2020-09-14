# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:51:49 2020

@author: Natan Steinbruch
"""

import process_dataset as pdt
import funcao_ativacao as fa
import cv2
#
#Bloco Principal
#

path = "C:\\Users\\natst\\OneDrive\\Natan Steinbruch\\IA-do-Ramo\\DataSet\\"
#Modifique o path para onde está a sua pasta DataSet

train,test,dataSet,train_saidas,test_saidas = pdt.process_data_set(path)
img = cv2.imread(path+"train/without_mask/83.jpg",0)
img = cv2.resize(img,(28,28)) / 255
img = img.flatten()
