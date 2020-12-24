# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 02:02:23 2020

@author: Natan Steinbruch
"""

import cv2
import os
import numpy as np

dimx = 28
dimy = 28

def process_dataset_folder(path_with_folder,lista_img,train_or_test,with_without,dataset,dataset_saidas):
    
    for name_img in lista_img:
        img = cv2.imread(path_with_folder + name_img, 0)
        img = cv2.resize(img,(dimx,dimy)) / 255
            
        dataset.append(img)
        dataset_saidas.append([with_without])
    
    print('Processamento concluido em %s'%(path_with_folder))

def process_data_set (path):
   
    dataset_saidas = []
    dataset = []

    
    lista_train_with_mask = os.listdir(path+"train\\with_mask")
    lista_train_without_mask = os.listdir(path+"train\\without_mask")
    lista_test_with_mask = os.listdir(path+"test\\with_mask")
    lista_test_without_mask = os.listdir(path+"test\\without_mask")
    process_dataset_folder(path+"train\\without_mask\\",lista_train_without_mask,"train",0,dataset,dataset_saidas)
    process_dataset_folder(path+"train\\with_mask\\",lista_train_with_mask,"train",1,dataset,dataset_saidas)
    process_dataset_folder(path+"test\\with_mask\\",lista_test_with_mask,"test",1,dataset,dataset_saidas)
    process_dataset_folder(path+"test\\without_mask\\",lista_test_without_mask,"test",0,dataset,dataset_saidas)
    
    print("DataSet pronto!!")
    return np.reshape(dataset,(len(dataset),dimx*dimy)),np.reshape(dataset_saidas,(len(dataset_saidas),1)),dimx,dimy
