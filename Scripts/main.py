# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:51:49 2020

@author: Natan Steinbruch
"""

import process_dataset as pdt
import cv2
#
#Bloco Principal
#

path = "C:\\Users\\natst\\OneDrive\\Natan Steinbruch\\IA-do-Ramo\\DataSet\\"
#Modifique o path para onde está a sua pasta DataSet

train,test,dataSet,train_saidas,test_saidas = pdt.process_data_set(path)

cv2.imshow("",train[0])
cv2.waitKey(0)
cv2.destroyAllWindows()