# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:50:20 2020

@author: Natan Steinbruch
"""
import numpy as np

def soma(e,p):
    return e.dot(p)
#dot = produto escalar

def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0
#
# Bloco Principal
#
entradas = np.array([-1,7,5])
pesos = np.array([0.8,0.1,0])
s = soma(entradas,pesos)
r = stepFunction(s)


