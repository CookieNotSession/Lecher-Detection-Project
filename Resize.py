# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from PIL import Image
#==============================================================================
# im = Image.open('obamaf.jpg')
# width = 19
# height = 19
# nim = im.resize((width,height), Image.BILINEAR)
# print(nim.size)
# nim.save('1919obamaf.jpg')
# 
#==============================================================================


def resizeandsave(img):
    im = Image.open(img)
    width = 30
    height = 30
    nim = im.resize((width,height), Image.BILINEAR)
    nim.save('new'+img)


def resize(img):
    im = Image.open(img)
    width = 30
    height = 30
    nim = im.resize((width,height), Image.BILINEAR)
    return nim


import numpy as np
import os
from PIL import Image
from skimage import io
from sklearn.neural_network import MLPClassifier

pic = io.imread('dada.jpg',as_grey=True)    
#newimg=resize(pic)
