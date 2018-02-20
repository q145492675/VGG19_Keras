# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:40:18 2018

@author: Herman Wu
"""

import os
import zipfile 

import Image_processing
import VGG19

if __name__=='__main__':
    Base_dir = os.path.dirname(__file__)
    Datazip_dir=Base_dir+'/dataset.zip'
    print('/*******************************************************/')
    print('         Extract and loading the data.........  ')
    fz=zipfile.ZipFile(Datazip_dir,'r')
    for file in fz.namelist():
        fz.extract(file,Base_dir)
###########################################################################################################
    DataBase_dir=Base_dir+'/dataset'
    Input_X,Input_Y=Image_processing.image_process(DataBase_dir)
    print(' \n           Finished!!  \n ')
    print('/*******************************************************/')
###########################################################################################################
    Model=VGG19.main(Input_X,Input_Y)