#!/usr/bin/env python
# coding: utf-8

# In[3]:


#segmentation lung cancer
from __future__ import division
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from array import *
from skimage import measure
import math
import numpy as np
import skfuzzy
def get_roi(img):
    a=img
    img= cv2.resize(img, (512,512))
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    a = cv2.resize(a, (512,512))
    mask = a.copy()
    mask1 = a.copy()
    m  = a.shape[0]
    n  = a.shape[1]
    for i in range (m):
        for j in range (n):
            z= math.sqrt((i-m/2)**2+(j-n/2)**2)
            if z <=150 :
                mask [i][j]=255
            else :
                mask [i][j]=0  
    for i in range (m):
        for j in range (n):
            if i>=130 and i<=380 :
                mask1 [i][j]=255
            else :
                mask1 [i][j]=0 
    ret, bw_img = cv2.threshold(a,127,256,cv2.THRESH_BINARY)
    bw_img1=bw_img
    for i in range (m):
        for j in range (n):
            bw_img1[i][j] = 255-bw_img[i][j]
            if mask[i][j] > 0 and bw_img1[i][j] > 0 and mask1[i][j]>0:
                bw_img[i][j] = 255 
            else : 
                bw_img[i][j] = 0
    bw_img  = bw_img1
    label = measure.label((bw_img1))
    propsa = measure.regionprops(label)
    x = []
    for i in propsa:
        x.append(i.area)
    x = np.asarray(x)
    for i in range(len(propsa)):
        j=i+1
        for j in range(len(propsa)):
            if x[i]>x[j]:
                x1   = x[j]
                x[j] = x[i]
                x[i] = x1
    u = 0
    v = 0
    for i in range(len(x)-1):
        z=x[i]-x[i+1]
        j=i+1
        if (x[i]>=2500)and(x[i]<=40000):
            if(x[i+1]>=0)and(x[i+1]<=40000):
                if (z<=30000)and(z>=500):
                    u=x[i]
                    v=x[i+1]
                    break
    y1=0
    y2=0
    for i in propsa:
        if i.area== u:
            y1=i.label
    for i in propsa:
        if i.area== v:
            y2=i.label
    BW=bw_img
    for i in range (m):
        for j in range (n):
            if label[i][j]==y1 or label [i][j]==y2:
                BW[i][j]=bw_img[i][j]
            else :
                BW[i][j]=0
    kernel  = np.ones((9,9),np.uint8)
    BW = cv2.dilate(BW,kernel,iterations = 1)
    kernel1  = np.ones((7,7),np.uint8)
    BW = cv2.dilate(BW,kernel1,iterations = 1)
    im_floodfill = BW.copy()
    h, w = BW.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = BW | im_floodfill_inv
    BW1=BW
    for i in range (m):
        for j in range (n):
            if BW [i][j] > 0 :
                BW1[i][j]=a[i][j]
            else :
                BW1[i][j]=0
    BW1 = BW1/255
    imdata = np.reshape(BW1, ( 1,262144))
    o = skfuzzy.cluster.cmeans(imdata, 3, 2, 0.0001, 1000000, init=None, seed=None)
    nn=o[1]
    N1=[]
    N2=[]
    N3=[]
    j  = o[1].shape[0]
    h  = o[1].shape[1]
    for i in range (j):
        for j in range (h):
            if i == 0:
                N1.append(nn[0][j])
            if i == 1:
                N2.append(nn[1][j])
            if i == 2:
                N3.append(nn[2][j])
    N1 = np.asarray(N1)
    j  = N1.shape[0]
    c1 = sum(N1)
    c1 = c1 / j
    N2 = np.asarray(N2)
    j  = N1.shape[0]
    c2 = sum(N2)
    c2 = c2 / j
    N3 = np.asarray(N3)
    j  = N1.shape[0]
    c3 =sum(N3)
    c3 = c3 / j
    X1 = []
    X2 = []
    X3 = []
    if (c1>c2) and (c1>c3):
        for i in range (j):
            X1.append(N1[i])
        if (c2>c3):
            for i in range (j):
                X2.append(N2[i])
            for i in range (j):
                X3.append(N3[i])
        else :
            for i in range (j):
                X2.append(N3[i])
            for i in range (j):
                X3.append(N2[i]) 
    if (c2>c1) and (c2>c3):
        for i in range (j):
            X1.append(N2[i])
        if (c1>c3):
            for i in range (j):
                X2.append(N1[i])
            for i in range (j):
                X3.append(N3[i])
        else :
            for i in range (j):
                X2.append(N3[i])
            for i in range (j):
                X3.append(N1[i])
    if (c3>c2) and (c3>c1):
        for i in range (j):
            X1.append(N3[i])
        if (c2>c1):
            for i in range (j):
                X2.append(N2[i])
            for i in range (j):
                X3.append(N1[i])
        else :
            for i in range (j):
                X2.append(N1[i])
            for i in range (j):
                X3.append(N2[i])
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    X3 = np.asarray(X3)
    imdx1 = np.reshape(X1, (512, 512))
    imdx2 = np.reshape(X2, (512, 512))
    imdx3 = np.reshape(X3, (512, 512))
    kernel1   =   np.ones((5,5),np.uint8)
    kernel2   =   np.ones((3,4),np.uint8)
    kernel3   =   np.ones((4,3),np.uint8)

    imdx4 = imdx3
    for i in range (m):
        for j in range (n):
            if imdx3[i][j]<0.5:
                imdx3[i][j]=0
            if imdx3[i][j]>0.5:
                imdx3[i][j]=255
    kernel4   =   np.ones((3,3),np.uint8)
    imdx3     =   cv2.morphologyEx(imdx3, cv2.MORPH_CLOSE, kernel4)
    imdx4     =   cv2.morphologyEx(imdx3, cv2.MORPH_OPEN, kernel1)
    imdx5     =   cv2.erode(imdx3,kernel2,iterations = 1)
    imdx6     =   cv2.erode(imdx5,kernel3,iterations = 1)
    imdx7     =   cv2.morphologyEx(imdx6, cv2.MORPH_OPEN, kernel4)
    labele  =   measure.label((imdx7))
    propsa1 = measure.regionprops(labele)
    z  = []
    z1 = []
    z2 = []
    for i in propsa1:
        z.append(i.area)
        z1.append(i.major_axis_length)
        z2.append(i.minor_axis_length)
        l=i
    z  = np.asarray(z)
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    y  = []
    for i in propsa1:
        y.append(i.label)
    y = np.asarray(y)
    mm=0.0
    float(mm)
    for i in range (len(propsa1)):
        mm = z2[i] / z1[i]
        if mm >0.50:
            if z[i] < 1000 :
                if z[i] > 40:
                    y1= y[i] 
                    break
            continue
        continue
    B1=a.copy()
    for i in range (m):
        for j in range (n):
            if labele[i][j]==y1 :
                B1[i][j]=imdx3[i][j]
            else :
                B1[i][j]=0
    kernel    =   np.ones((10,10),np.uint8)
    B1        = cv2.dilate(B1,kernel,iterations = 1)
    for i in range (m):
        for j in range (n):
            if B1[i][j] > 0 :
                B1[i][j]=a[i][j]
            else :
                B1[i][j]=0
    for i in range (m):
        for j in range (n):
            if B1[i][j] > 0 :
                B1[i][j]=imdx3[i][j]
            else :
                B1[i][j]=0
    for i in range(1,B1.shape[0]-1):
        for j in range(1,B1.shape[1]-1):
            if (B1[i,j]>0):
                img[i,j,2]=255
                img[i,j,1]=0
                img[i,j,0]=0
                
    cv2.imshow('Cancer',img)
    cv2.waitKey(0)
    return img

