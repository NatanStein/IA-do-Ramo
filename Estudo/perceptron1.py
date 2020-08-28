# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:33:20 2020

@author: Natan Steinbruch
"""
def soma(e,p):
    soma = 0
    for i in range(3):
        soma += e[i] * p[i]
    return soma

def stepFunction(soma):
    if(soma > 0):
        return 1
    return 0
#
# Bloco Principal
#
entradas = [-1,7,5]
pesos = [0.8,0.1,0]
s = soma(entradas,pesos)
r = stepFunction(s)


