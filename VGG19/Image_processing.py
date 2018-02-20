# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:20:56 2018

@author: Herman Wu
"""

import cv2
import numpy as np
import skimage.io as io

def loading(data_dir,Type):
    temp=data_dir+'/img_'+Type+'_*.jpg'
    coll=io.ImageCollection(temp)
    return coll

def pre_process(img):
    BLACK = [0,0,0]
    top,bottom,left,right=(0,0,0,0)
    length=img.shape[0]
    width=img.shape[1]
    longest_edge=max(length,width)
    if length < longest_edge:
        temp=longest_edge-length
        top=temp//2
        bottom=temp-top
    elif width<longest_edge:
        temp=longest_edge-width
        left=temp//2
        right=temp-left
    else: 
        pass
    constant=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
    return constant

def merge(data_1,data_2):
    length1=range(len(data_1))
    length2=range(len(data_2))
    temp=np.zeros([len(data_1)+len(data_2),64,64,3]).astype(np.float32)
    for i in length1:
        img=data_1[i]/255
        img=pre_process(img)
        res=cv2.resize(img,(64,64))
        temp[i,:,:,:]=res
    for i in length2:
        img=data_2[i]/255
        img=pre_process(img)
        res=cv2.resize(img,(64,64))
        temp[(i+len(data_1)),:,:,:]=res
    return temp

def label(data_1,data_2):
    len1=len(data_1)
    len2=len(data_2)
    temp=np.ones([len1+len2,1]).astype(np.float32)
    for i in range(len1):
        temp[i]=0
    return temp

def image_process(data_dir):
    c_image=loading(data_dir,'c')
    b_image=loading(data_dir,'b')
    InputX=merge(c_image,b_image)
    InputY=label(c_image,b_image)
    return InputX,InputY