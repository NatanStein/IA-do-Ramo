# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 02:02:23 2020

@author: Natan Steinbruch
"""


import cv2
import os
import numpy as np

def process_dataset_folder(path_with_folder,lista_img,train_or_test,with_without,train,test,dataset,train_saidas,test_saidas):
        
    for name_img in lista_img:
        
        img = cv2.imread(path_with_folder + name_img, 0)
        img = cv2.resize(img,(28,28))
        img = cv2.threshold(img,140,255,cv2.THRESH_BINARY)
        if train_or_test == "train":
            train.append(np.where(img[1]==255,1,img[1]))
            train_saidas.append(with_without)
        elif train_or_test == "test":
            test.append(np.where(img[1]==255,1,img[1]))
            test_saidas.append(with_without)
        dataset.append([np.where(img[1]==255,1,img[1]),with_without])
    
def process_data_set (path):
   
    train = []
    test = []
    train_saidas = []
    test_saidas = []
    dataset = []
    
    lista_train_with_mask = os.listdir(path+"train\\with_mask")
    lista_train_without_mask = os.listdir(path+"train\\without_mask")
    lista_test_with_mask = os.listdir(path+"test\\with_mask")
    lista_test_without_mask = os.listdir(path+"test\\without_mask")
    
    process_dataset_folder(path+"train\\without_mask\\",lista_train_without_mask,"train",0,train,test,dataset,train_saidas,test_saidas)
    process_dataset_folder(path+"train\\with_mask\\",lista_train_with_mask,"train",1,train,test,dataset,train_saidas,test_saidas)
    process_dataset_folder(path+"test\\with_mask\\",lista_test_with_mask,"test",1,train,test,dataset,train_saidas,test_saidas)
    process_dataset_folder(path+"test\\without_mask\\",lista_test_without_mask,"test",0,train,test,dataset,train_saidas,test_saidas)
    
    return train,test,dataset,train_saidas,test_saidas
