# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:56:49 2020

@author: Natan Steinbruch
"""

import numpy as np

#
#

def sigmoid (x):
    return 1.0/ (1.0 + np.exp(-x))

def derivada_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#
#
    
def relu(x):
    return np.where(x<0,0,x)

def derivada_relu(x):
   return np.where(x<=0,0,1)

#
#
       
def leaky_relu(x):
    return np.where(x<=0,0.01*x,x)
 
def derivada_leaky_relu(x):
   return np.where(x<0,0.01,1)

