# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 22:02:58 2020

@author: Natan Steinbruch
"""

import numpy as np

def mae (y,y_pred,t,derivative=False):
    if derivative:
        return np.where(y_pred > y, 1, -1) / t
    return np.mean(np.abs(y-y_pred))

def mse (y,y_pred,t,derivative=False):
    if derivative:
        return -(y-y_pred) / t
    return 0.5 * np.mean((y-y_pred)**2)

def bce (y, y_pred,t,derivative=False):
    if derivative:
        return -(y - y_pred) / (y_pred * (1-y_pred) * t)
    return -np.mean(y*np.log(abs(y-y_pred)) + (1 - y)*np.log(abs(1 - y_pred)))
    