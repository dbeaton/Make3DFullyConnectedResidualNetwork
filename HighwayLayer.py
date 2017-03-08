import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d



def InnerHighwayLayer(img_in,Num_Filters=32):
    #1 conv + relu
    #2 conv
    Y1=convolution2d(inputs=img_in,num_outputs=Num_Filters,stride=1,kernel_size=(3,3),padding='SAME')#automatically does relu
    #Need to find out if these auto handle biases
    Y2=convolution2d(inputs=Y1,num_outputs=Num_Filters,stride=1,kernel_size=(3,3),activation_fn=None,padding='SAME')    
    return Y2


def HwBlock(img_in,Num_Filters=32):
    #only need to call this function
    #need lambda0,1,2
    
    lambda0=tf.Variable([1.0])
    lambda1=tf.Variable([1.0])
    lambda2=tf.Variable([1.0])
    #initializing highway multiplier scalars
    
    Yi=InnerHighwayLayer(img_in,Num_Filters)#intermediate output from first
    #inner highway block
    Y1=tf.add(tf.multiply(lambda1,img_in),Yi)
    Yi2=InnerHighwayLayer(Y1,Num_Filters)
    Y2=tf.add(tf.add(tf.multiply(lambda0,img_in),tf.multiply(lambda2,Y1)),Yi2)
    
    return Y2 # this function call comprises one total Outer Lambda-residualblock

    