# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:29:45 2017

@author: admin
"""

from sklearn.neural_network import MLPClassifier
import os
from PIL import Image#讀圖的
import numpy as np



os.chdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\30X30 Aface')
filelist=os.listdir()
x=np.zeros((len(filelist),30*30,4))

#開圖讀圖，把圖轉換
for i in range(len(filelist)):
    IMG=Image.open(filelist[i])
    x[i,:]=np.array(IMG.getdata())#當張照片放到x[i]中
A_Aface=x.copy()

os.chdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\30X30 nonA')
filelist=os.listdir()
x=np.zeros((len(filelist),30*30,4))
#開圖讀圖，把圖轉換
for i in range(len(filelist)):
    IMG=Image.open(filelist[i])
    x[i,:]=np.array(IMG.getdata())#當張照片放到x[i]中
nonA_Aface=x.copy()

os.chdir('..')

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

A_Aface=rgb2gray(A_Aface)
nonA_Aface=rgb2gray(nonA_Aface)



X=np.append(A_Aface,nonA_Aface,axis=0)#把兩個np的array上下接在一起
X=(X-np.mean(X))/np.std(X)
y=np.append(np.ones((A_Aface.shape[0],1)),np.zeros((nonA_Aface.shape[0],1)),axis=0)#做答案

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X,y)

y2=clf.predict(X)

#==============================================================================
# print(np.sum(y[:,0]==y2)/y.shape[0])#test 的正確率
#==============================================================================

def resize(img):
    im = Image.open(img)
    width = 30
    height = 30
    nim = im.resize((width,height), Image.BILINEAR)
    return nim


def aaornonaa(img):
    img=np.array(resize(img))
    img=rgb2gray(img).flatten()
    y2=clf.predict(img)
    return y2
    


im = Image.open('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\upload_file_python\\images\\new2.PNG')



#im = Image.open('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\dada.jpg')
#print(aaornonaa('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\upload_file_python\images\\new18.PNG'))



