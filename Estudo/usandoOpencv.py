# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:53:43 2020

@author: Vinícius
"""
# Importação das bibliotecas
import numpy as np
import cv2
from matplotlib import pyplot as plt

#Função para plotagem de uma imagem utilizando Opencv e matplotlib; 
def showImage(img):
    #Transforma a orientação de BGR para RGB, evitando que a imagem fique azulada.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

#Função que obtém e retorna os valores BGR de um pixel.    
def getColor(img, y, x):
    return img.item(y, x, 0), img.item(y,x,1), img.item(y, x, 2)

#Função que altera os valores BGR de um pixel.
def setColor(img, b, g, r, y, x):
    img.itemset((y, x, 0), b)
    img.itemset((y, x, 1), g)
    img.itemset((y, x, 2), r)
    
    return img

def main():
    #Leitura da imagem, parâmetro "0" a transforma em grayscale.
    obj_img = cv2.imread("imgs/82-with-mask.jpg")
    altura, largura, canais = obj_img.shape
    
    for y in range(altura):
        for x in range(largura):
            #Pegando informações de azul, verde e vermelho de cada pixel da imagem.
            
            # azul = obj_img.item(y, x, 0)
            # verde = obj_img.item(y, x, 1)
            # vermelho = obj_img.item(y, x, 2)
            
            #Pegando essas informações de cada pixel através de uma função.
            azul, verde, vermelho = getColor(obj_img, y, x)
            
            #Trocando os valores verde e vermelho por valores nulos, deixando apenas o azul em cada pixel.
            # obj_img.itemset((y, x, 1), 0)
            # obj_img.itemset((y, x, 2), 0)
            
            #Trocando os valores verde e azul por valores nulos, deixando apenas o vermelho anteriormente adquirido pela função getColor
            #Tudo isso através de uma nova função
            # obj_img = setColor(obj_img, 0, 0, vermelho, y, x)
            
            #Recortando uma área específica da imagem.
            mask_img = obj_img[80: 80 + 60, 50: 50 + 100]
    
    cv2.imwrite("82-with-mask-red.jpg", obj_img)
    #Chamar a função para plotagem da imagem.
    showImage(obj_img)
    #Chamando a função para a plotagem da área recortada da imagem.
    showImage(mask_img)
    
main()  