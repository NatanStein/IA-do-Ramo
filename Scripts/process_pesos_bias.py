# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:25:47 2020

@author: Natan Steinbruch
"""


import numpy as np
import funcao_ativacao as fa
import os
import cv2

path_bias ='C:/Users/natst/OneDrive/Natan Steinbruch/IA-do-Ramo/Promissores/Bias-NNN-SIG-MSE-0.5-3000-0.3' 
path_pesos = 'C:/Users/natst/OneDrive/Natan Steinbruch/IA-do-Ramo/Promissores/Pesos-NNN-SIG-MSE-0.5-3000-0.3'
#Modifique seu PATH


def read_weights_bias(path_bias, path_pesos):
    
    w_layer1 = np.genfromtxt(path_pesos +'/Pesos1.txt', delimiter= ',')
    w_layer2 = np.genfromtxt(path_pesos +'/Pesos2.txt', delimiter=',')
    w_layer3 = np.genfromtxt(path_pesos +'/Pesos3.txt', delimiter=',')
    w_layer4 = np.genfromtxt(path_pesos +'/Pesos4.txt', delimiter=',')
    
    b_layer1 = np.genfromtxt(path_bias +'/Bias1.txt', delimiter= ',')
    b_layer2 = np.genfromtxt(path_bias +'/Bias2.txt', delimiter=',')
    b_layer3 = np.genfromtxt(path_bias +'/Bias3.txt', delimiter=',')
    b_layer4 = np.genfromtxt(path_bias +'/Bias4.txt', delimiter=',')
    
    return w_layer1.reshape((784,53)), w_layer2.reshape((53,36)), w_layer3.reshape((36,25)), w_layer4.reshape((25,1)), b_layer1.reshape((1,53)), b_layer2.reshape((1,36)), b_layer3.reshape((1,25)), b_layer4.reshape((1,1))


def predict (test,funcao_ativacao):
    
    if funcao_ativacao == "sigmoid":
        f = fa.sigmoid
    elif funcao_ativacao == "relu":
        f = fa.relu
    elif funcao_ativacao == "leaky_relu":
        f = fa.leaky_relu
        
    camada_oculta1 = f(np.dot(test,w_layer1) + b_layer1)
    camada_oculta2 = f(np.dot(camada_oculta1,w_layer2) + b_layer2)
    camada_oculta3 = f(np.dot(camada_oculta2,w_layer3) + b_layer3)

    return f(np.dot(camada_oculta3,w_layer4) + b_layer4)

def process_entradas(path):
    
    entradas = []
    lista_img = os.listdir(path)
    
    for name_img in lista_img:
        img = cv2.imread(path+name_img, 0)
        img = cv2.resize(img,(28,28)) / 255
        entradas.append(img.flatten().reshape((1,28*28)))
        
    return np.reshape(entradas,(len(lista_img),784)), lista_img

#
#Bloco Principal
#

w_layer1, w_layer2, w_layer3, w_layer4, b_layer1, b_layer2, b_layer3, b_layer4 = read_weights_bias(path_bias, path_pesos)

entradas,lista_img = process_entradas('C:\\Users\\natst\\OneDrive\\Natan Steinbruch\\IA-do-Ramo\\Validacao\\')

saida = predict(entradas,"sigmoid")

for resp,nome in zip(saida,lista_img):
    print('Nome da imagem: %s  Resposta: %f'%(nome,resp))

#saidas = np.where(saida >= 0.5,1,0)
#print(np.count_nonzero(saidas == 1)/len(saidas) * 100)