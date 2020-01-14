# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:54:13 2017

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
class_id = x[c] ## classes
class_id[class_id == 1] = 0 ## binary classification
class_id[class_id > 1] = 1
y = x.drop(c,1)

train = y[y.index < int(n/2)] ### half of the records as training set
test = y[y.index >= int(n/2)] ### half of the records as test set
train_class = class_id[class_id.index < n//2]
test_class = class_id[class_id.index >= n//2]

HIDDEN_LAYER = c//2 ### number of hidden layer neurons
GRADIENT = 0.05 ### gradient used for optimizer
ITERATIONS = 5000
#input placeholders
x1 = tf.placeholder(tf.float32,shape = (None,c), name = "input")
y1 = tf.placeholder(tf.float32,shape = (None) , name = "classes")

# weight and bias tensors for layer 1
w1 = tf.Variable(tf.truncated_normal(shape =[c,HIDDEN_LAYER], mean = 0.0, stddev = 1.0, dtype =tf.float32 ))
b1 = tf.Variable(tf.constant(0.1,shape=[HIDDEN_LAYER]))

layer1 = tf.add(tf.matmul(x1,w1),b1)
layer1 = tf.nn.sigmoid(layer1)

# weight and bias tensors for layer 2 -- output layer
w2 = tf.Variable(tf.truncated_normal(shape =[HIDDEN_LAYER,1], mean = 0.0, stddev = 1.0, dtype =tf.float32 ))
b2 = tf.Variable(tf.constant(0.1,shape=[1]))

## output results
out = tf.add(tf.matmul(layer1,w2), b2)
out = tf.reshape(out,[-1])
y_cal = tf.nn.sigmoid(out)

## cross entropy cost function used to optimize
entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y1, logits = out )
cost = tf.reduce_mean(entropy)

## gradient descent algorithm used for minimization of cost
Optimized = tf.train.GradientDescentOptimizer(GRADIENT).minimize(cost)

## output neurons containing values less than 0.5 are classified as 0
## others are classified as 1
y_check = tf.where(tf.less(y_cal,tf.fill([n//2],0.5)), tf.zeros([n//2]),tf.ones([n//2]))

## calculation of true positive, true negative, false positive and false
## negative for sentivity and specificity calculations
t_p = tf.logical_and(tf.equal(y_check,y1),tf.equal(y_check,tf.ones(y_check.shape)))
True_positive = tf.reduce_sum( tf.cast(t_p, tf.float32) )

f_p = tf.logical_and(tf.not_equal(y_check,y1),tf.equal(y_check,tf.ones(y_check.shape)))
False_positive = tf.reduce_sum( tf.cast(f_p, tf.float32) )

t_n = tf.logical_and(tf.equal(y_check,y1),tf.equal(y_check,tf.zeros(y_check.shape))) 
True_negative = tf.reduce_sum( tf.cast(t_n, tf.float32) )

f_n = tf.logical_and(tf.not_equal(y_check,y1),tf.equal(y_check,tf.zeros(y_check.shape)))
False_negative = tf.reduce_sum( tf.cast(f_n, tf.float32) )

Sensitivity = True_positive/(True_positive + False_positive)
Specificity = True_negative/(True_negative + False_negative)

## session initializations
sess = tf.InteractiveSession()

## intiialization of all variables
sess.run(tf.global_variables_initializer())

cost_vect = np.zeros([ITERATIONS])
Sens_vect = np.zeros([ITERATIONS])
Spec_vect = np.zeros([ITERATIONS])
## the algorithms is iterated and trained on the training set 
## and the cost, specificity and sensitivity is calculated for each cycle
for i in np.arange(ITERATIONS):
    feed = {x1 : train, y1: train_class }
    sess.run(Optimized,feed_dict=feed)
    cost_vect[i] = cost.eval(feed_dict =feed,session = sess)
    Spec_vect[i] = Specificity.eval(feed_dict =feed,session = sess)
    Sens_vect[i] = Sensitivity.eval(feed_dict =feed,session = sess)
    if(i%100 == 0):
        print("Turn: %d\n\tCost: %f\n\tSpecificity: %f\n\tSensitivity: %f" % (i,cost_vect[i],Spec_vect[i],Sens_vect[i]))
## plotting of results
def PLOT(title,cost_vect):
    plt.figure()
    plt.title('Binary classification ' + title )
    plt.xlabel('iteration number')
    plt.ylabel(title)
    plt.plot(np.linspace(0,ITERATIONS-1,num=ITERATIONS),cost_vect)
    plt.show()

PLOT('cost',cost_vect)
PLOT('Sensitivity',Sens_vect)
PLOT('Specificity',Spec_vect)
## results evaluated on the test and training set
print("Training set:\n\tSensitivity %f\n\tSpecificity %f" % (Sensitivity.eval(feed,sess),Specificity.eval(feed,sess)) )
feed = {x1 : test, y1: test_class }
print("Test set:\n\tSensitivity %f\n\tSpecificity %f" % (Sensitivity.eval(feed,sess),Specificity.eval(feed,sess)) )
    
sess.close()