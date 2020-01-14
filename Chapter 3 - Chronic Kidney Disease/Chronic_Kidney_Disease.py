# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:18:51 2017

@author: Rehan Rajput
"""
## the file chronic_kidney_disease.csv must be in the same directory as
## this code
import numpy as np
import pandas as pd
import re
import sklearn.tree as sk

feat = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc' ,'rbcc','htn','dm','cad','appet','pe','ane','class']
data = pd.read_csv("chronic_kidney_disease.csv",skiprows=29,header = None,na_values = ['?','\t','\t?'],names=feat)

data.wbcc.map(lambda x: x[2:] if '\t' in str(x) else x)

#Method 1: uncomment the line below to apply method 1 and make sure to comment the method 2.
data = data.dropna(axis=0)



data = data.set_index(np.arange(1,len(data)+1))

## normal is changed numerically to '1'
## the records whose names contain the expression 'no', would be classified as 0
## the records 'poor' would be changed to '0'
## any other alphabetical records are changed to '1'
##  --- regular expressions are used
rep = ['normal',re.compile('.*no.*'),'poor',re.compile('[a-zA-Z].*')]
val = [1,0,0,1]


data = data.replace(to_replace = rep,value = val)

# Method 2:uncomment the two lines below to apply method 2 and make sure to comment the method 1.
#for columns in data:
#    data[columns] = data[columns].fillna(int(data[columns].mean()))
    
    
y = data['class']
x = data.drop('class',1)
clf = sk.DecisionTreeClassifier("entropy")
clf = clf.fit(x,y)
sk.export_graphviz(clf,out_file="Tree.dot",feature_names=feat[:len(feat)-1],class_names = ['notckd','ckd'],filled=True,rounded=True,special_characters = True)
# use anaconda prompt and type 'dot Tree.dot -Tpng -o Tree.png
