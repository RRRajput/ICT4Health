# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:48:45 2017

@author: Rehan Rajput
"""
## the file arrhythmia.csv must be in the same directory as the code

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
class_id[class_id > 1] = 2 ## all the classes>1 are made into class '2' for binary classification
y = x.drop(c,1) 

y1 = y[class_id == 1]   ### records of healthy patients
y2 = y[class_id == 2]   ### records of unhealthy patients

x1 = y1.mean()  ## mean of healthy patients
x2 = y2.mean()  ## mean of unhealthy patients
T_p = 0 ## intiailizing True and false positives along with true and false negatives
F_p = 0
T_n = 0
F_n = 0
est_class_id = pd.Series(np.empty(n)) 
for i in y.T:
    diff1 = np.linalg.norm(x1 - y.T[i])
    diff2 = np.linalg.norm(x2 - y.T[i])
    if(diff1 < diff2):  ### using the minimun distance criterion
        est_class_id[i] = 1
        if (class_id[i] == 1):
            T_p=T_p+1
        else:
            F_p=F_p+1
    else:
        est_class_id[i] = 2
        if(class_id[i] == 1):
            F_n = F_n +1
        else:
            T_n = T_n +1
c = len(y)
T_p = T_p/(T_p + F_p)  ## TP rate
T_n = T_n/(T_n + F_n)   ## TN rate
Acc_MD= sum(est_class_id == class_id)/len(class_id) ## accuracy of this method
pi_1 = sum(class_id == 1)/len(class_id) ### prior of healthy patients
pi_2 = sum(class_id == 2)/len(class_id) ### prior of unhealthy patients
 
####### PCA of the data
R1 = pd.DataFrame(np.cov(y1.T),index = y1.columns,columns = y1.columns)
R2 = pd.DataFrame(np.cov(y2.T),index = y2.columns, columns = y2.columns)

def Reduction(R1):
    V1,M1 = np.linalg.eig(R1)
    M1 = pd.DataFrame(M1)
    V1 = pd.Series(V1)
    V1.sort_values(inplace=True,ascending=False)
    tot_sum = np.sum(np.abs(V1))
    for i in np.arange(1,len(V1)+1):
        if sum(np.abs(V1.head(i))) >= (1- 10**-6)*tot_sum:
            break
    V1 = V1.head(i)
    r_cols = set(M1.columns) - set(V1.index)
    M1 = M1.drop(r_cols,1)
    return V1,M1

V1,M1 = Reduction(R1)
V2,M2 = Reduction(R2)

z1 = y1.dot(M1)
z2 = y2.dot(M2)
######## covariance matrices are diagonal for z1 and z2. this can be checked by:
######## np.cov(z1.T)    
w1 = z1.mean()
w2 = z2.mean()

##### Bayes criterion with pdf function
def getPDF(y,u):
    R1 = pd.DataFrame(np.cov(y.T),index = y.columns , columns = y.columns)
    x1 = y.mean()
    diff = u - x1
    R1_inv = pd.DataFrame(np.linalg.inv(R1), R1.index,R1.columns)
    
    num = (-1/2) * diff.T.dot(R1_inv.dot(diff))
    den = 1/2*(len(y.columns)*np.log(2*np.pi) + np.log(np.linalg.det(R1)))
    return num - den

s1 = y.dot(M1)
s2 = y.dot(M2)

T_p1 = 0
T_n1 = 0
F_p1 = 0
F_n1 = 0
est_class_id1 = pd.Series(np.empty(n))
prob1 = pd.Series(np.zeros(n))
prob2 = pd.Series(np.zeros(n))
for i in y.T:
    prob1[i] = np.log(pi_1) + getPDF(z1,s1.T[i])
    prob2[i] = np.log(pi_2) + getPDF(z2,s2.T[i])
    if(prob1[i] > prob2[i]): ### Bayes Criterion
        est_class_id1[i] = 1
        if (class_id[i] == 1):
            T_p1=T_p1+1
        else:
            F_p1=F_p1+1
    else:
        est_class_id1[i] = 2
        if(class_id[i] == 1):
            F_n1 = F_n1 +1
        else:
            T_n1 = T_n1 +1
T_p1 = T_p1/(T_p1 + F_p1) ## True positive rate
T_n1 = T_n1/(T_n1 + F_n1) ## true negative rate
Acc_Prob= sum(est_class_id1 == class_id)/len(class_id) ##accuracy of this method