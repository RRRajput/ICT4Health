# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:52:36 2017

@author: Rehan Rajput
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

x = pd.read_csv("arrhythmia.csv",names = np.arange(0,280),na_values = ['?'])
## records containing 'na' values are dropped
x = x.dropna(axis=1)
## columns (features) containing no values are dropped
x = x.drop(x.columns[x.apply(lambda c: sum(c==0) >= len(x))],axis=1)
## rearrangement of column names
x = x.T.set_index(np.arange(0,len(x.T))).T


n = len(x) # total patients
c = len(x.columns) - 1 # total features
class_id = x[c] ## class vector
y = x.drop(c,1)
x = pd.DataFrame(np.zeros((c,16)),np.arange(0,c),np.arange(1,17))
## x stores the nx16 matrix which signifies the class for each record
for i in np.arange(1,17):
    x[i] = y[class_id ==i].mean() ## mean of records of each class
x = x.fillna(0) ## all the NaN values to be replaced with 0
est_class_id = pd.Series(np.empty(n))
diff = pd.DataFrame(np.zeros((n,16)),columns =x.columns )

for i in y.T:
    for j in x:
        diff[j][i]= np.linalg.norm(x[j] - y.T[i]) ## distance from the center
                                                ## of each class
    est_class_id[i] = diff.T[i].argmin()
Acc_MD= sum(est_class_id == class_id)/len(class_id) ## accuracy

conf_mat = np.zeros([16,16])
## confusion matrix build up
for i in np.arange(len(est_class_id)):
    conf_mat[int(est_class_id[i]-1)][int(class_id[i]-1)] += 1

conf_mat = conf_mat.T/[x if x>0 else 1 for x in np.sum(conf_mat,axis=1)]
conf_mat = conf_mat.T