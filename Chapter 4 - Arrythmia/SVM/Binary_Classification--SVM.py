# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 19:15:29 2018

@author: Rehan Rajput
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

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


#### Pre-processing data using PCA'
Ry = (1/len(y))*y.T.dot(y)
A,U = np.linalg.eig(Ry)
A = pd.Series(A,index = Ry.index)
U = pd.DataFrame(U, index = Ry.index, columns = Ry.columns)

def L_features(A):
    P = sum(A)
    A = A.sort_values(ascending = False)
    for i in range(3,len(A)+1):
        temp = A.head(i)
        L = sum(temp)
        if L >= (1- 10**-6)*P:
            return temp
    return temp

A = L_features(A)

def minimize_U(U,A):
    min_set =  set(U.columns) - set(A.index)
    for i in min_set:
        U = U.drop(i,1)
    return U

U = minimize_U(U,A)
y = U.T.dot(y.T).T


train = y[y.index < int(n/2)] ## dividing the data into test and training set
test = y[y.index >= int(n/2)]
train_class = class_id[class_id.index < n//2]
test_class = class_id[class_id.index >= n//2]

ITER = 5
acc = np.zeros(ITER)
log__ = [10,100,1000,10000,100000]
### SVM applied on a number of 'C' values
for i in range(ITER):
    clf = svm.SVC(C= pow(log__[i],-1))
    clf.fit(train,train_class)
    acc[i] = clf.score(test,test_class)


    