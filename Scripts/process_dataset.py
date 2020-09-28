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
def process_dataset_folder(path_with_folder,lista_img,train_or_test,with_without,train,test,dataset,train_saidas,test_saidas):
    
    for name_img in lista_img:
        
        img = cv2.imread(path_with_folder + name_img, 0)
        img = cv2.resize(img,(dimx,dimy)) / 255
        #img = cv2.threshold(img,140,255,cv2.THRESH_BINARY)
        #train.append([np.where(img[1]==255,1,img[1]),with_without])
        if train_or_test == "train":
            train.append(img.flatten().reshape((1,dimx*dimy)))
            train_saidas.append(with_without)
        elif train_or_test == "test":
            test.append(img.flatten().reshape((1,dimx*dimy)))
            test_saidas.append(with_without)
        dataset.append([img,with_without])


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
    
    print("DataSet pronto!!")
    return np.reshape(train,(616,dimx*dimy)),np.reshape(test,(198,dimx*dimy)),dataset,np.reshape(train_saidas,(616,1)),np.reshape(test_saidas,(198,1)),dimx,dimy
