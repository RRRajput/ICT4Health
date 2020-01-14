# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:31:51 2017

@author: Rehan Rajput
"""
import numpy as np



## this function receives a vector with the index number of the darkest pixels
## present on the imaginary vertical line drawn in the middle of the image
## and makes sure there are no gaps between the dark pixels
## it finally returns the first and the last index of the 
def IE_Points(v):
    for i in np.arange(0,len(v) - 1):
        if(v[i + 1] - v[i] > 4):
            return IE_Points(v[i + 1:])
        if(v[i] - v[i + 1] > 1):
            return IE_Points(v[:i])
    return v[0],v[-1]


#test_labels = np.copy(new_labels)
# test_labels are the labels
# ind is the index number of the darkest centroid
# border is the label number for the border pixels
def MyContour(test_labels,ind,border):
    ## the initial y index is in the middle of the image
    y_ini =int( test_labels.shape[1]/2 )
    ## example contains the index numbers of the dark pixels on the imaginary
    ## vertical line drawn on the middle of the image
    example = np.nonzero(np.transpose(test_labels)[y_ini,:] == ind)[0]
    x_final,x_ini = IE_Points(example)
    ## initial direction is '1' which indicates movement upwards from the left-side
    direction = 1
    x = x_ini
    y = y_ini
    ## if direction = '1', the the trace checks around from the left side and
    ## tries to move upwards
    ## if the direction = '-1' the trace checks from the right side and tries to
    ## go downward
    ## in both the cases the checks are carried out clockwise
    while(True):
        test_labels[x,y] = border
        ## the while loop runs until the initial and final position are the same
        if(x==x_ini and y==y_ini and direction ==  - 1):
            break
        ## the direction is changed once the trace reaches the top of the mole
        ## the direction is made '-1' which means movement downwards
        if (x==x_final):
            direction= -1*direction
        if(x == x_ini and direction == -1):
            y = y + 1*direction
        elif (test_labels[x,y-1*direction] == ind):
            y = y - 1*direction
        elif( test_labels[x - 1*direction,y - 1*direction] == ind):
            x = x - 1*direction
            y = y - 1*direction
        elif( test_labels[x - 1*direction,y] == ind):
            x = x - 1*direction
        elif( test_labels[x - 1*direction,y + 1*direction] == ind):
            x = x - 1*direction
            y = y + 1*direction
        elif( test_labels[x,y + 1*direction] == ind):
            y = y + 1*direction
        elif( test_labels[x + 1*direction,y + 1*direction] == ind):
            x = x + 1*direction
            y = y + 1*direction
        elif( test_labels[x + 1*direction,y] == ind):
            x = x + 1*direction
        elif( test_labels[x + 1*direction,y - 1*direction] == ind):
            x = x + 1*direction
            y = y - 1*direction
        else:
            y = y + 1*direction
        if ( x > x_ini):
            x_ini = x
