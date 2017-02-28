import numpy as np
from os import listdir,getcwd
import cv2 as cv
import scipy.io as sp
import matplotlib.pyplot as plt



def GetBatch(batch_size,size=(1200,1200)):
#loads make3D dataset from file
    
    Dir=getcwd()
    
    TrainPath='/Train400Img/'
    TrainImgs=listdir(Dir+TrainPath)
    
    LabelsPath='/Train400Depth/'
    LabelImgs=listdir(Dir+LabelsPath)

    

    T=[]
    L=[]
    
    for icount in range((batch_size)):
    #Scanning through and loading all images, both training
    #and labels, Will crop then convert to numpy array and return
    #tmpT=Image.open(Dir+TrainPath+TrainImgs[icount])
    #Dont Make BatchSIze too high, severely slows down performance
        tmpL=sp.loadmat(Dir+LabelsPath+LabelImgs[icount])
        ## I DONT KNOW WHAT THESE MAT FILES MEAN????
        tmpT=cv.imread(Dir+TrainPath+TrainImgs[icount])
        tmpT=cv.resize(tmpT,size,tmpT,interpolation=cv.INTER_CUBIC)
       
                
        T.append(np.array(tmpT,dtype='float32'))
        L.append(tmpL)
        
    T=np.array(T,dtype='float32')
    
    
    # I THINK ITS IN BGR NOT RGB
    return T,L
#NEED TO GET MAT FILES

T,L=GetBatch(5) ## Just a test call