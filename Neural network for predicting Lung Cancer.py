#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.models import load_model
#from imgaug import augmenters as iaa
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D,Convolution2D
from sklearn.model_selection import train_test_split
import random
#import pickle
import pandas as pd
import cv2
import os
#import skimage
 
#from keras.callbacks import LearningRateScheduler, ModelCheckpoint
 
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)

def do_dataset(x,y):
    my_img=[]
    my_label=[]
    for i in range(x):
        my_img.append('x{}.jpg'.format((i+1)))
        my_label.append(0)
    for i in range(y):
        my_img.append('y{}.jpg'.format((i+1)))
        my_label.append(1)
    return my_img,my_label

my_img,my_label=do_dataset(212,204)

my_series=pd.Series(data=my_img)
data=pd.DataFrame(my_series)
data['label']=my_label
data

def load_img_label(datadir, df):
    image_path = []
    label = []
    for i in range(len(df)):
        indexed_data = data.iloc[i]
        img = indexed_data[0]
        image_path.append(os.path.join(datadir, img.strip()))
        label.append(indexed_data['label'])
    image_paths = np.asarray(image_path)
    label = np.asarray(label)
    return image_paths, label

def img_preprocess(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #imgcv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (155, 155))
    img = img[40:120,40:120]
    img = img/255
    return img

def data_gen(image_paths, label_ang):

    batch_img = []
    batch_label = []
    
    for i in range(len(image_paths)):
      im=mpimg.imread(image_paths[i])
      im=img_preprocess(im)  
      im = np.expand_dims(im, axis=-1)
      labell =label[i]
      batch_img.append(im)
      batch_label.append(labell)
      A=np.asarray(batch_img)
      B=np.asarray(batch_label)
    return A,B

image_paths, label = load_img_label('' , data)
xx_train_gen, yy_train_gen = data_gen(image_paths, label)

def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(80, 80, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
  
    model.compile(Adam(lr = 0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = modified_model()
print(model.summary())
history = model.fit(xx_train_gen, yy_train_gen, validation_split=0.05, epochs =18, batch_size =20, verbose = 1, shuffle = 1)

