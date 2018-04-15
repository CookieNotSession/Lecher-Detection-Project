# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:12:14 2018

@author: HIM_LAB
"""

#from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import face_recognition
import os
from PIL import Image
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


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

A_Aface=rgb2gray(A_Aface)
nonA_Aface=rgb2gray(nonA_Aface)

X=np.append(A_Aface,nonA_Aface,axis=0)#把兩個np的array上下接在一起
X=(X-np.mean(X))/np.std(X)
y=np.append(np.ones((A_Aface.shape[0],1)),np.zeros((nonA_Aface.shape[0],1)),axis=0)#做答案
y=y.ravel()

'''fetch face'''
def fetchface(IMG):
    os.chdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject')
    image = face_recognition.load_image_file("C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\"+IMG) #這邊隨意放入一張要測試的照片
    position = face_recognition.face_locations(image)[0]
    top, right, bottom, left = position
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    #resize
    out=pil_image.resize((30,30), Image.BILINEAR)
    out.save('face'+IMG)


fetchface('test.jpg')

#==============================================================================
# '''test data'''
# testIMG = Image.open('out.jpg')
# testIMG = np.asarray(testIMG)
# testIMG = testIMG[:,:,0]
# testIMG = testIMG.reshape(1, -1)
#==============================================================================


#==============================================================================
# '''SVM'''
# clf = SVC()
# clf.fit(X, y)
# #y2 = clf.predict(X)
# y3 = clf.predict(testIMG)
# print(y3)
#==============================================================================
