# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 02:02:23 2020

@author: Natan Steinbruch
"""


import cv2

imagem = cv2.imread("Screenshot_3.png")
cv2.imshow("My image",imagem)

cv2.waitKey(0)
cv2.destroyAllWindows()