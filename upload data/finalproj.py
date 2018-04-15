#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:26:56 2017

@author: cookiehiker
"""

import os
from flask import Flask, request, render_template, send_from_directory
from sklearn.neural_network import MLPClassifier
from PIL import Image#讀圖的
import numpy as np
import face_recognition
from skimage import io


#做model
#========================================================================


trainface = []
os.chdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\30X30 Aface')
filelist = os.listdir()
for i in filelist:
    picture = io.imread(i,as_grey=True)
    picture = np.ndarray.flatten(picture)
    trainface.append(picture)
trainface = np.asanyarray(trainface)
   
trainnonface = []
os.chdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\30X30 nonA')
filelist = os.listdir()
for i in filelist:
    picture = io.imread(i,as_grey=True)
    picture = np.ndarray.flatten(picture)
    trainnonface.append(picture)
trainnonface = np.asanyarray(trainnonface)


X = np.append(trainface,trainnonface,axis=0) #把trainface與trainnonface兩array上下接在一起
y = np.append(np.ones((trainface.shape[0],1)),np.zeros((trainnonface.shape[0],1)),axis=0) #一矩陣放trainface
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)   
print('okokokokok')
os.chdir('..')

#========================================================================





#========================================================================
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

def fetchface(IMG,dirn):
    os.chdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\upload_file_python\\images\\'+dirn)
    image = face_recognition.load_image_file("C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\upload_file_python\\images\\"+dirn+"\\"+IMG) #這邊隨意放入一張要測試的照片
     
    if(len(face_recognition.face_locations(image))):
        position = face_recognition.face_locations(image)[0]
        top, right, bottom, left = position
        face_image = image[top:bottom, left:right]
    else:face_image = image[:,:]
    pil_image = Image.fromarray(face_image)
    #resize
    out=pil_image.resize((30,30), Image.BILINEAR)
    out.save('face'+IMG)
    return
    


dirnum=1

#========================================================================

__author__ = 'ibininja'

app = Flask(__name__)



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    print('rrrrrrrrrrrrrrrrrrrrrrrrr')
    for upload in request.files.getlist("file"):
#        dirname=upload.filename[0:-4]
        global dirnum
        print(dirnum ,"in")
        dirname=str(dirnum)
        dirnum+=1
    print('rrrrrrrrrrrrrrrrrrrrrrrrr')
    
    target = os.path.join(APP_ROOT, 'images/'+dirname+'/')
#    print(target)
    if not os.path.isdir(target):os.mkdir(target)
#    else:
#        print("Couldn't create upload directory: {}".format(target))
#    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
#        print(upload)
#        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
#        print ("Save it to:", destination)
        upload.save(destination)
        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
        fetchface(filename,dirname)
        testans=aaornonaa('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\upload_file_python\\images\\'+dirname+'\\face'+filename)
        print("測試結果:",testans)
        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
        showpic=''
        if(testans==0):showpic='nice.jpg'
        else:showpic='bad.jpg'
        facepic='face'+filename
        
    return render_template("result.html", image_name=facepic, aaf=showpic, dirname=dirname)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)



@app.route('/gallery/<dirnn>')
def get_gallery(dirnn):
    image_names = os.listdir('C:\\Users\\admin\\Desktop\\智慧型\\finalproject\\upload_file_python\\images\\'+dirnn)
    print(image_names)
    return render_template("gallery.html", image_names=image_names)



@app.route('/isaa/<dirnn>')
def pop2aatrain(dirnn):
    print('跑進來拉拉拉拉拉')
    return render_template("isaa.html", dirnn=dirnn)
#---------------------------以下test------------------------------

@app.route('/testtttttttt')
def testtttttttt():
    return render_template("jinjatest.html", name='AAAAAAAAAAAA',sss='123123123123123')

@app.route ( '/user/<username>' ) 
def  show_user_profile ( username ): 
    # show the user profile for that user 
    return  'User %s '  %  username

@app.route ( '/post/<int:post_id>' ) 
def  show_post ( post_id ): 
    # show the post with the given id, the id is an integer 
    return  'Post %d '  %  post_id

@app.route ( '/jinjatest/' ) 
@app.route ( '/jinjatest/<name>' ) 
def  hello ( name = None ): 
    return  render_template ( 'jinjatest.html' ,  name = name )

#---------------------------以上test------------------------------


if __name__ == "__main__":
    app.run(host='0.0.0.0')