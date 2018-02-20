# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:29:52 2018

@author: Herman Wu
"""

import numpy as np  
np.random.seed(1337)  # for reproducibility  
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import  MaxPooling2D  
from keras import optimizers
from keras.utils import np_utils   
from keras.layers import Conv2D
import pandas as pd
import time
from keras.models import load_model


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("The running time of this code: %s " % self.elapsed(time.time() - self.start_time) )


def ModelBuild(Model,input_shape):
    Model.add(Conv2D(64,(3,3),padding='same',input_shape=input_shape,name="VGG19_Conv1"))
    Model.add(Activation('relu'))
    Model.add(Conv2D(64,(3,3),padding='same',name="VGG19_Conv2"))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D(pool_size=(2,2),name="VGG19_Pool1"))
###########################################################################################################   
    Model.add(Conv2D(128,(3,3),padding='same',name="VGG19_Conv3"))
    Model.add(Activation('relu'))    
    Model.add(Conv2D(128,(3,3),padding='same',name="VGG19_Conv4"))
    Model.add(Activation('relu'))    
    Model.add(MaxPooling2D(pool_size=(2,2),name="VGG19_Pool2"))
########################################################################################################### 
    '''
    Model.add(Conv2D(256,(3,3),padding='same',name="VGG19_Conv5"))
    Model.add(Activation('relu'))    
    Model.add(Conv2D(256,(3,3),padding='same',name="VGG19_Conv6"))
    Model.add(Activation('relu')) 
    Model.add(Conv2D(256,(3,3),padding='same',name="VGG19_Conv7"))
    Model.add(Activation('relu'))    
    Model.add(Conv2D(256,(3,3),padding='same',name="VGG19_Conv8"))
    Model.add(Activation('relu')) 
    Model.add(MaxPooling2D(pool_size=(2,2),name="VGG19_Pool3"))
########################################################################################################### 
    Model.add(Conv2D(512,(3,3),padding='same',name="VGG19_Conv9"))
    Model.add(Activation('relu'))       
    Model.add(Conv2D(512,(3,3),padding='same',name="VGG19_Conv10"))
    Model.add(Activation('relu'))
    Model.add(Conv2D(512,(3,3),padding='same',name="VGG19_Conv11"))
    Model.add(Activation('relu'))
    Model.add(Conv2D(512,(3,3),padding='same',name="VGG19_Conv12"))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D(pool_size=(2,2),name="VGG19_Pool4"))    
########################################################################################################### 
    Model.add(Conv2D(512,(3,3),padding='same',name="VGG19_Conv13"))
    Model.add(Activation('relu'))       
    Model.add(Conv2D(512,(3,3),padding='same',name="VGG19_Conv14"))
    Model.add(Activation('relu'))
    Model.add(Conv2D(512,(3,3),padding='same',name="VGG19_Conv15"))
    Model.add(Activation('relu'))
    Model.add(Conv2D(512,(3,3),padding='same',name="VGG19_Conv16"))
    Model.add(Activation('relu'))
    '''
########################################################################################################### 
    Model.add(Flatten())
    Model.add(Dense(4096,name="VGG19_Dense1"))
    Model.add(Activation('relu'))
    Model.add(Dropout(0.5))
    Model.add(Dense(4096,name="VGG19_Dense2"))
    Model.add(Activation('relu'))
    Model.add(Dropout(0.5))
    Model.add(Dense(2,name="VGG19_OutLay"))
    Model.add(Activation('sigmoid'))
########################################################################################################### 

def configure(Model,Loss='binary_crossentropy'):
    optimizers.adadelta(lr=0.01,decay=2e-4)
    Model.compile(loss=Loss,optimizer='adadelta',metrics=['accuracy'])
    print('\n################    The Detail of the VGG19     ###################')    
    print(Model.summary())
    time.sleep(5)
    print('\n######################################################################\n')

def main(Docx,DocY,epoch=20,batch_size=50,nb_classes=2):
    img_rows=Docx.shape[1] 
    img_cols=Docx.shape[2] 
    in_shape= (img_rows, img_cols, 3)  
    Y_train = np_utils.to_categorical(DocY, nb_classes)
########################################################################################################### 
    VGG19=Sequential()
    ModelBuild(VGG19,in_shape)
    configure(VGG19)
########################################################################################################### 
    timer = ElapsedTimer()        
    print('/*******************************************************/\n')
    print(' Now we begin to training VGG19 model.\n')
    print('/*******************************************************/\n') 
    VGG19.fit(Docx,Y_train,batch_size=batch_size,epochs=epoch,shuffle=True,validation_split=0.10) 
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
    print('/*******************************************************/\n')    
    return VGG19