# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:51:49 2020

@author: Natan Steinbruch
"""

import process_dataset as pdt
import funcao_ativacao as fa
import funcao_custo as fc
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def test_predict(teste,teste_saidas,pesos1,pesos2,pesos3,pesos4,bias1,bias2,bias3,bias4,prt_saida=False):
    inp1 = np.dot(teste,pesos1) + bias1
    camada_oculta1 = fa.sigmoid(inp1)

    inp2 = np.dot(camada_oculta1,pesos2) + bias2
    camada_oculta2 = fa.sigmoid(inp2)

    inp3 = np.dot(camada_oculta2,pesos3) + bias3
    camada_oculta3 = fa.sigmoid(inp3)

    inp4 = np.dot(camada_oculta3,pesos4) + bias4
    camada_saida = fa.sigmoid(inp4)

    custo = fc.binary_cross_entropy(teste_saidas,camada_saida,qtt_test,False)
    
    if prt_saida:
        print("erro: %.10f"%(custo))
        print(camada_saida)
    
    return custo
#
#Bloco Principal
#

path = "C:\\Users\\natst\\OneDrive\\Natan Steinbruch\\IA-do-Ramo\\DataSet\\"
#Modifique o path para onde está a sua pasta DataSet

dataset,dataset_saidas,dimx,dimy = pdt.process_data_set(path)

train, test, train_saidas, test_saidas = train_test_split(dataset,dataset_saidas,test_size=0.3)

n_camada_oculta1 = 53//1
n_camada_oculta2 = 36//1
n_camada_oculta3 = 25//1

pesos1 = 2 * np.random.random((dimx*dimy, n_camada_oculta1)) -1
pesos2 = 2 * np.random.random((n_camada_oculta1, n_camada_oculta2)) -1
pesos3 = 2 * np.random.random((n_camada_oculta2, n_camada_oculta3)) -1
pesos4 = 2 * np.random.random((n_camada_oculta3, 1)) -1

bias1 = np.zeros((1,n_camada_oculta1))
bias2 = np.zeros((1,n_camada_oculta2))
bias3 = np.zeros((1,n_camada_oculta3))
bias4 = np.zeros((1,1))

momentum = 0.01  
prev_dw1 = 0.0
prev_dw2 = 0.0
prev_dw3 = 0.0
prev_dw4 = 0.0

qtt_treino = len(train)
qtt_test = len(test)
dinamic = False
epochs = 20000
learning_rate = 0.3
erros =[]
erros2 = []

for epocas in range(epochs+1):
    
    inp1 = np.dot(train,pesos1) + bias1
    camada_oculta1 = fa.sigmoid(inp1)

    inp2 = np.dot(camada_oculta1,pesos2) + bias2
    camada_oculta2 = fa.sigmoid(inp2)

    inp3 = np.dot(camada_oculta2,pesos3) + bias3
    camada_oculta3 = fa.sigmoid(inp3)

    inp4 = np.dot(camada_oculta3,pesos4) + bias4
    camada_saida = fa.sigmoid(inp4)
    #camada_saida = np.where(camada_saida > 0,1,0)

    custo = fc.binary_cross_entropy(train_saidas,camada_saida,qtt_treino,False)
    result = test_predict(test,test_saidas,pesos1,pesos2,pesos3,pesos4,bias1,bias2,bias3,bias4)
    erros.append(custo)
    erros2.append(result)
    print("epoca: %d/%d erro_train: %f erro_test: %f"%(epocas,epochs,custo,result))
    
    derivada_saida = fc.binary_cross_entropy(train_saidas,camada_saida,qtt_treino,True)
    
    dinp4 = fa.derivada_sigmoid(inp4) * derivada_saida # dy =  da(y) * df(y')
    derivada_oculta3 = np.dot(dinp4,pesos4.T) # dx = w.T * dy 
    d_pesos4 = np.dot(dinp4.T,camada_oculta3) # dw = x * dy.T
    d_bias4 = 1.0 * dinp4.sum(axis=0,keepdims=True) # db = sum(dy) 
    
    dinp3 = fa.derivada_sigmoid(inp3) * derivada_oculta3
    derivada_oculta2 = np.dot(dinp3,pesos3.T)
    d_pesos3 = np.dot(dinp3.T,camada_oculta2)
    d_bias3 = 1.0 * dinp3.sum(axis=0,keepdims=True)
    
    dinp2 = fa.derivada_sigmoid(inp2) * derivada_oculta2
    derivada_oculta1 = np.dot(dinp2,pesos2.T)
    d_pesos2 = np.dot(dinp2.T,camada_oculta1)
    d_bias2 = 1.0 * dinp2.sum(axis=0,keepdims=True)
    
    dinp1 = fa.derivada_sigmoid(inp1) * derivada_oculta1
    derivada_entrada = np.dot(dinp1,pesos1.T)
    d_pesos1 = np.dot(dinp1.T,train)
    d_bias1 = 1.0 * dinp1.sum(axis=0,keepdims=True)
    
    pesos4 = pesos4 + (-learning_rate * d_pesos4.T + momentum*prev_dw4)
    pesos3 = pesos3 + (-learning_rate * d_pesos3.T + momentum*prev_dw3)
    pesos2 = pesos2 + (-learning_rate * d_pesos2.T + momentum*prev_dw2)
    pesos1 = pesos1 + (-learning_rate * d_pesos1.T + momentum*prev_dw1)
    
    bias4 = bias4 - learning_rate * d_bias4
    bias3 = bias3 - learning_rate * d_bias3
    bias2 = bias2 - learning_rate * d_bias2
    bias1 = bias1 - learning_rate * d_bias1
    
    prev_dw1 = d_pesos1.T
    prev_dw2 = d_pesos2.T
    prev_dw3 = d_pesos3.T
    prev_dw4 = d_pesos4.T
    
    if dinamic:
        plt.ion()
        plt.cla()
        plt.clf()
        plt.plot(erros,label="train")
        plt.plot(erros2, label="test")
        plt.legend()
        plt.pause(0.01)

result = test_predict(test,test_saidas,pesos1,pesos2,pesos3,pesos4,bias1,bias2,bias3,bias4)


if dinamic == False:
    plt.plot(erros,label="train")
    plt.plot(erros2, label="test")
    plt.legend()
    plt.show()
else:
    plt.ioff()

print(result)
resposta = input("Deseja fazer um Dump dos pesos e bias? S/N: ")

if resposta == "S":
    
    np.savetxt(path[0:-8]+"Pesos\\pesos1.txt",pesos1, delimiter=",")
    np.savetxt(path[0:-8]+"Pesos\\pesos2.txt",pesos2, delimiter=",")
    np.savetxt(path[0:-8]+"Pesos\\pesos3.txt",pesos3, delimiter=",")
    np.savetxt(path[0:-8]+"Pesos\\pesos4.txt",pesos4, delimiter=",")
    
    np.savetxt(path[0:-8]+"Bias\\Bias1.txt",bias1, delimiter=",")
    np.savetxt(path[0:-8]+"Bias\\Bias2.txt",bias2, delimiter=",")
    np.savetxt(path[0:-8]+"Bias\\Bias3.txt",bias3, delimiter=",")
    np.savetxt(path[0:-8]+"Bias\\Bias4.txt",bias4, delimiter=",") 
    
    print("Dump dos pesos e bias concluido")

else:
    print("Pesos e Bias não foram salvos")
    