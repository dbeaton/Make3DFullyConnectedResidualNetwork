import tensorflow as tf
import numpy as np
import HighwayLayer as hw

#all of this is just for the quick mnist test will be deleted
from keras.datasets.cifar10 import load_data

data=load_data()
Xtrain=data[0][0]
Ytrain=data[0][1]

Xtest=data[1][0]
Ytest=data[1][1]

del data
###########################################################################


#need scaling then 
#Need to find out the number of filters needed for each conv
 #RUNNING quick mnist test


X=tf.placeholder(tf.float32,[None,32,32,3])
Y=hw.HwBlock(X,Num_Filters=3)



# Everything past this point is just for a quick test run to see if the model would even run
# will totally be changed as this runs on Mnist but its a good layout for how
# to implement training, losses etc
#
#Need to implement method to read cost values, save the graph to tensorboard,
#saving model and other metric.

#add prediction function 
#as simple as sess.run(Y,feed_dict{X:TestData})
mse=tf.losses.mean_squared_error(X,Y)
optimizer=tf.train.GradientDescentOptimizer(.01).minimize(mse)

def trainModel(Batch,epochs=10):
    
    sess=tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for step in range(epochs):
        sess.run(optimizer,feed_dict={X:Batch})
def Predict(testdata):
    sess=tf.InteractiveSession()
    Out=sess.run(Y,feed_dict={X:testdata})
    return Out
