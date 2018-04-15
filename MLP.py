#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:44:34 2017

@author: cookie040667
"""

import numpy as np
import os
from PIL import Image
from skimage import io


trainface = []
os.chdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\30X30 Aface')
filelist = os.listdir()
for i in range(len(filelist)):
    i = i+1
    picture = io.imread('new'+str(i)+'.png',as_grey=True)
    picture = np.ndarray.flatten(picture)
    trainface.append(picture)
trainface = np.asanyarray(trainface)
   
trainnonface = []
os.chdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\30X30 nonA')
filelist = os.listdir()
for i in range(len(filelist)):
    i = i+1
    picture = io.imread('new'+str(i)+'.png',as_grey=True)
    picture = np.ndarray.flatten(picture)
    trainnonface.append(picture)
trainnonface = np.asanyarray(trainnonface)


from sklearn.neural_network import MLPClassifier
X = np.append(trainface,trainnonface,axis=0) #把trainface與trainnonface兩array上下接在一起
y = np.append(np.ones((trainface.shape[0],1)),np.zeros((trainnonface.shape[0],1)),axis=0) #一矩陣放trainface
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)   

os.chdir('..')


#++++++++++++++++++++++++++++++++++++
def resize(img):
    im = Image.open(img)
    width = 30
    height = 30
    nim = im.resize((width,height), Image.BILINEAR)
    return nim

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def aaornonaa(img):
    pic=np.array(np.array(resize(img)))
    pic=np.ndarray.flatten(rgb2gray(pic))
    testpic = []
    testpic = np.asanyarray(pic)
    testpic = np.asanyarray(testpic) 
    testpic = np.reshape(testpic, (1, -1))
    y2 = clf.predict(testpic)
    return y2



ans=aaornonaa('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\30X30 nonA\\new11.PNG')










#print(aaornonaa('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\upload_file_python\\images\\new2.PNG'))


#==============================================================================
# testpic = []
# pic = io.imread('.\\30X30 Aface\\new2.PNG',as_grey=False)
# pic=rgb2gray(pic)/255
# 
# 
# pic = np.ndarray.flatten(pic)
# testpic = np.asanyarray(pic)
# testpic = np.asanyarray(testpic) 
# testpic = np.reshape(testpic, (1, -1))
# 
# 
# Ty2 = clf.predict(testpic)
# print(Ty2)
#==============================================================================

