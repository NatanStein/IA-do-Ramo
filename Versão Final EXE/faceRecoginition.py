# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:58:03 2020

@author: Natan Steinbruch
"""
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pip"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])

import cv2
import numpy as np

def sigmoid (x):
    return 1.0/ (1.0 + np.exp(-x))

def read_weights_bias(div):
    
    w_layer1 = np.genfromtxt('Pesos1.txt', delimiter= ',')
    w_layer2 = np.genfromtxt('Pesos2.txt', delimiter=',')
    w_layer3 = np.genfromtxt('Pesos3.txt', delimiter=',')
    w_layer4 = np.genfromtxt('Pesos4.txt', delimiter=',')
    
    b_layer1 = np.genfromtxt('Bias1.txt', delimiter= ',')
    b_layer2 = np.genfromtxt('Bias2.txt', delimiter=',')
    b_layer3 = np.genfromtxt('Bias3.txt', delimiter=',')
    b_layer4 = np.genfromtxt('Bias4.txt', delimiter=',')
    
    return w_layer1.reshape((784,53//div)), w_layer2.reshape((53//div,36//div)), w_layer3.reshape((36//div,25//div)), w_layer4.reshape((25//div,1)), b_layer1.reshape((1,53//div)), b_layer2.reshape((1,36//div)), b_layer3.reshape((1,25//div)), b_layer4.reshape((1,1))


def faceRecoginition():
    faces = []
    cam = cv2.VideoCapture(0)
    while (True): 
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            faces = clf.detectMultiScale(gray)
            for x, y, w, h in faces:
                image = gray[y-60: y+h+100, x-100: x+w+100]
                image = cv2.resize(image,(28,28)) / 255
                image = image.flatten().reshape((1,784))
                camada_oculta1 = sigmoid(np.dot(image,w_layer1) + b_layer1)
                camada_oculta2 = sigmoid(np.dot(camada_oculta1,w_layer2) + b_layer2)
                camada_oculta3 = sigmoid(np.dot(camada_oculta2,w_layer3) + b_layer3)
                val = sigmoid(np.dot(camada_oculta3,w_layer4) + b_layer4)[0]
                if (val < 0.5):
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                else:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
                cv2.imshow('video', frame) #mostra o vídeo
        except Exception as e:
            print(e)
            pass
    
        if cv2.waitKey(115) & 0xFF == ord('s'):
            break
        
    cam.release()
    cv2.destroyAllWindows() #fecha as janelas


clf = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
w_layer1, w_layer2, w_layer3, w_layer4, b_layer1, b_layer2, b_layer3, b_layer4 = read_weights_bias(1)
print('''
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄       ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄       ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌     ▐░░▌▐░░░░░░░░░░░▌
 ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌   ▐░▐░▌▐░█▀▀▀▀▀▀▀█░▌
     ▐░▌     ▐░▌       ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌
     ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌ ▐░▐░▌ ▐░▌▐░▌       ▐░▌
     ▐░▌     ▐░░░░░░░░░░░▌     ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌▐░▌       ▐░▌
     ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌     ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░█▀▀▀▀█░█▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌   ▀   ▐░▌▐░▌       ▐░▌
     ▐░▌     ▐░▌       ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌  ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌
 ▄▄▄▄█░█▄▄▄▄ ▐░▌       ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌     ▐░▌      ▐░▌ ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌
▐░░░░░░░░░░░▌▐░▌       ▐░▌     ▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌     ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀       ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀       ▀         ▀  ▀         ▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀ 
 ''')
print('''
      Coders: Natan Steinbruch
              Vinícius L. Santos
              Gabriel R. O. Camargo
              Nicholas C. Villela
              Andre R. Xavier
              ''')
print("Github: https://github.com/NatanStein/IA-do-Ramo")
faceRecoginition()
