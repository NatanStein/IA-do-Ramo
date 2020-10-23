# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 00:49:19 2020

@author: natst
"""


import imageio

path = 'C:\\Users\\natst\\OneDrive\\Natan Steinbruch\\dataset\\Test\\'
images = []
for i in range(5000):
    images.append(imageio.imread(path+'plot'+str(i)+'.png'))
imageio.mimsave(path+'movie.gif', images, format="GIF", duration=1/50)