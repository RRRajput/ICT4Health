# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:23:15 2018

@author: Rehan Rajput
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv("arrhythmia.csv",names = np.arange(0,280),na_values = ['?'])
## records containing 'na' values are dropped
x = x.dropna(axis=1)
## columns (features) containing no values are dropped
x = x.drop(x.columns[x.apply(lambda c: sum(c==0) >= len(x))],axis=1)
## rearrangement of column names
x = x.T.set_index(np.arange(0,len(x.T))).T


n = len(x) # total patients
c = len(x.columns) - 1 # total features
class_1 = x[c] - 1
class_id= np.zeros([n,16])
for i in np.arange(n): ## class matrix containing nx16 dimensions
    class_id[i,int(class_1[i])] = 1
y = x.drop(c,1)
class_id = pd.DataFrame(class_id)
train = y[y.index < int(n/2)] ## training set
test = y[y.index >= int(n/2)] ## test set
train_class = class_id[class_id.index < n//2] ## class sets
test_class = class_id[class_id.index >= n//2]

HIDDEN_LAYER = c//2 ## hidden layer neurons
GRADIENT = 0.7  ## gradient used for gradient descent function
ITERATIONS = 3000
## input placeholders
x1 = tf.placeholder(tf.float32,shape = (None,c), name = "input")
y1 = tf.placeholder(tf.float32,shape = (None,16) , name = "classes")

# layer 1 tensors
w1 = tf.Variable(tf.truncated_normal(shape =[c,HIDDEN_LAYER], dtype =tf.float32 ))
b1 = tf.Variable(tf.constant(0.1,shape=[HIDDEN_LAYER]))

layer1 = tf.matmul(x1,w1) + b1
layer1 = tf.nn.sigmoid(layer1)

# layer 2 tensors -- output tensors
w2 = tf.Variable(tf.truncated_normal(shape =[HIDDEN_LAYER,16], dtype =tf.float32 ))
b2 = tf.Variable(tf.constant(0.1,shape=[16]))

out = tf.matmul(layer1,w2) + b2
y_cal = tf.nn.sigmoid(out)

# cost function calculated using cross entropy cost function
entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y1, logits = out)
cost = tf.reduce_mean(entropy)

## gradient descent algorithm for optimization
Optimized = tf.train.GradientDescentOptimizer(GRADIENT).minimize(cost)

## calculating the accuracy
correct = tf.equal(tf.argmax(y1,1),tf.argmax(out,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

## making the confusion matrix
conf_mat = tf.confusion_matrix(labels = tf.argmax(y1,1), predictions=tf.argmax(out,1))
sess = tf.InteractiveSession()

# intialization of variables
sess.run(tf.global_variables_initializer())
cost_vect = np.zeros([ITERATIONS])
acc_vect = np.zeros([ITERATIONS])
## the algorithm is iterated and trained on the training set and the results
## are calculated
for i in np.arange(ITERATIONS):
    feed = {x1 : train, y1: train_class}
    sess.run(Optimized,feed_dict=feed)
    cost_vect[i] = cost.eval(feed_dict =feed,session = sess)
    acc_vect[i] = accuracy.eval(feed,sess)
    if(i%100 == 0):
        print("Turn: %d\n\tCost: %f\n\tAccuracy: %f" % (i,cost_vect[i],acc_vect[i]))
## plots the results
def PLOT(title,cost_vect):
    plt.figure()
    plt.title('16-Class classification ' + title )
    plt.xlabel('iteration number')
    plt.ylabel(title)
    plt.plot(np.linspace(0,ITERATIONS-1,num=ITERATIONS),cost_vect)
    plt.show()

PLOT('cost',cost_vect)
PLOT('Accuracy',acc_vect)

print("Cost on the test ",cost.eval(feed_dict = {x1:test,y1:test_class},session= sess))
print("Accuracy on the test ",accuracy.eval(feed_dict = {x1:test,y1:test_class},session= sess))
## computing the normalized confusion matrix
def Compute_Conf_Mat(feed,sess):
    c_mat = conf_mat.eval(feed_dict = feed, session = sess)
    sum_vect = [x if x>0 else 1 for x in np.sum(c_mat,axis=1)]
    c_mat = c_mat.T/sum_vect
    return c_mat.T

Confusion_Matrix_Train = Compute_Conf_Mat(feed,sess)
Confusion_Matrix_Test = Compute_Conf_Mat(feed = {x1:test,y1:test_class},sess=sess)
sess.close()
