# got the code from https://github.com/RubensZimbres/Repo-2017/blob/master/Face%20Recognition%20Autoencoder

from __future__ import print_function
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.core import Reshape
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
from keras.optimizers import SGD

# 'haarcascade_frontalface_default.xml', if it will not work directly then copy the path of the file inside opencv folder(download from github) as it has been done below
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('/home/nikhil/Documents/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
print(face_cascade)
img = cv2.imread('3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# print(faces[1])
# print(faces.shape[0])

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
#     # cv2.imshow('img',img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

a=[]
for i in range(0,faces.shape[0]):
    a.append(gray[faces[i][1]:faces[i][1]+faces[i][3],faces[i][0]:faces[i][0]+faces[i][2]])
    
import matplotlib.pyplot as plt
plt.imshow(a[1],cmap=plt.get_cmap('gray'))

for k in range(0,faces.shape[0]):    
    print(a[k].shape)

import scipy.misc

img1=[]
img2=[]
for i in range(0,faces.shape[0]):
    scipy.misc.imsave('face{}.jpg'.format(i), a[i])
    # img1.append(cv2.cvtColor(cv2.imread('face{}.jpg').format(i), cv2.COLOR_BGR2GRAY))
    img1.append(cv2.cvtColor(cv2.imread('face'+str(i)+'.jpg'), cv2.COLOR_BGR2GRAY))
    img2.append(cv2.resize(img1[i], (36, 36)))

img2=np.array(img2)
# print("--------------------")
# print(img2.shape)

for k in range(0,faces.shape[0]):    
    print(img2[k].shape)

batch_size = 30
nb_classes = 10
img_rows, img_cols = 36, 36
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape=(36,36,1)

learning_rate = 0.02
decay_rate = 5e-5
momentum = 0.9
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

recog = Sequential()
recog.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))
recog.add(UpSampling2D(size=(2, 2)))
recog.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))
recog.add(Convolution2D(20, 3, 3,init='glorot_uniform'))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))
recog.add(MaxPooling2D(pool_size=(3,3)))
recog.add(Convolution2D(6, 2, 2,init='glorot_uniform'))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))
recog.add(Convolution2D(6, 3, 3,init='glorot_uniform'))
recog.add(Activation('relu'))
recog.add(UpSampling2D(size=(2, 2)))
recog.add(Convolution2D(1, 1, 1,init='glorot_uniform'))
recog.add(Activation('relu'))

recog.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['accuracy'])
recog.summary()

recog.fit(img2.reshape(6,36,36,1), img2.reshape(6,36,36,1),
                nb_epoch=10,
                batch_size=30,verbose=1)
                
a=recog.predict(img2[0].reshape(1,36,36,1))
# print(a)
print(a.shape)
# scipy.misc.imsave('face.jpg', a)
# cv2.imshow('a', a)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
