# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:45:26 2018

@author: admin
"""

import os
from PIL import Image#讀圖的
import numpy as np

from sklearn.cluster import KMeans

import collections


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
#X=(X-np.mean(X))/np.std(X)
y=np.append(np.ones((A_Aface.shape[0],1)),np.zeros((nonA_Aface.shape[0],1)),axis=0)#做答案


kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
z=kmeans.labels_


AAfacecount=collections.Counter(z[0:277])
nonAAfacecount=collections.Counter(z[277:])

aafaceans=AAfacecount.most_common(1)[0][0]
nonAAfacecount=nonAAfacecount.most_common(1)[0][0]

training_accuracy=(np.sum(z[0:277]==aafaceans)+np.sum(z[277:]==nonAAfacecount))/len(z)

#--------------------------
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA(n_components=2)
pca.fit(X)
W=pca.components_
W=W.transpose()
#==============================================================================
# def PCATrain(D,R):#Ｄ：原始資料，Ｒ：reasure（要留多少維）
#     M,F=D.shape
#     meanv=np.mean(D,axis=0)#回傳1*50矩陣
#     D2=D-np.matlib.repmat(meanv,M,1)#create一個跟Ｄ一樣大的矩陣，每個得值都是全部的mean
#     C=np.dot(np.transpose(D2),D2)#C:50*50,C[i,j]代表ij間的變異程度(variation)
#     EValue,Evector=np.linalg.eig(C)
#     EV2=np.cumsum(EValue)/np.sum(EValue)#除以sum可以mapping到0~1之間
# #    累加到該個的資訊含量
#     num=np.where(EV2>=R)[0][0]+1
# #    num個
# #    meanv:zeromean的平均向量，
#     return meanv,Evector[:,range(num)]
# #meanv:N個vector
# 
# 
# meanv,W=PCATrain(X,0.4)
# 
# 
#==============================================================================

X = W  # we only take the first two features.
y= np.concatenate((np.ones(A_Aface.shape[0]),np.zeros(nonA_Aface.shape[0])),axis=0)
x_min, x_max = X[:, 0].min() - .01, X[:, 0].max() + .01
y_min, y_max = X[:, 1].min() - .01, X[:, 1].max() + .01

plt.figure(2, figsize=(8, 6))
plt.clf()



# Plot the training points
plt.scatter(X[0:277, 0], X[0:277, 1], color='r',edgecolor='k')
plt.scatter(X[277:, 0], X[277:, 1], color='b',edgecolor='k')
plt.xlabel('W1')
plt.ylabel('W2')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
print('training_accuracy=',training_accuracy)

