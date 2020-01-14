# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:36:07 2017

@author: Rehan Rajput
"""
### The pictures of the moles must be stored in the same directory as
### this file


import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

# This function takes a vector 'v' as an argument ('v' is usually a vector
# which contains the positions of the values satsifying certain condition)
# and a variable named 'center'.
# the main purpose of this function is to remove multiply regions, so that we only
# have one region, where the mole is.
def ReducedVector(v,center):
# runs a for-loop through the length of the vector 'v'
    for i in np.arange(0,len(v) - 1): 
# checks if two successive positions are separated by a distance bigger than 50 or not?
# which would signify that they're part of two different regions
        if (abs(v[i] - v[i+1]) > 50):
# if the two regions are separated by a distance of 50 pixels, we choose the one nearer
# to the center, since that's where the mole is
            if ( abs(v[i] - center) > abs(v[i+1] - center)):
                return np.array(v[i+1:]),False
            else:
                return np.array(v[:i]),False
# returns True, only if no furthur reduction is required (i.e: only one region)
    return v,True

# This function is used to find the starting and the ending index (on one axis)
# for the region inside of which, the mole can be found
    

def getIndices(labels,ind,threshold,distance,center):
# runs through the y-axis of the and sums all dark colored pixels in all rows
    v = sum(labels[:,:] == ind)
# only considers the rows where the sum is greater than the threshold
    v = np.nonzero(v >= threshold)[0]
# performs reduction to remove the presence of regions other than that for the mole
    complete_reduction = False
    while(complete_reduction == False):
        (v,complete_reduction) = ReducedVector(v,center)
# takes the starting and the end point
    start = v[0]
    end = v[-1]
# if the starting point is zero or the end point is the last one, by default 
# takes 150 and 300 as starting and end point.
# (happens in cases, where there are other dark regions spanning big part of the image)
    if (start <= 0):
        start = distance+150
    if (end + distance >= labels.shape[0]):
        end = 300 - distance
    return start-distance,end+distance
# this function is to find the rectangle containing the mole, once the median is found
def RectangleRange(labels,centro_x,ind,center):
# finds all the dark pixels on the line crossing through the median
    y_limits = np.nonzero(labels[centro_x,:] == ind)[0]
# performs reduction to just keep the region near the center of image (i.e: mole region)
    complete_reduction = False
    while(complete_reduction == False):
        y_limits,complete_reduction = ReducedVector(y_limits,center)
# takes the first and the last dark pixel on the line
    y_rect_min = y_limits[0]
    y_rect_max = y_limits[-1]
    return y_rect_min-25,y_rect_max+25

### Print the clustered image
def PrintLabels(labels,centroid,filename,num):
    img__ = np.zeros([labels.shape[0],labels.shape[1],3])
    for i in np.arange(labels.shape[0]):
        for j in np.arange(labels.shape[1]):
            img__[i,j,:] = centroid[labels[i,j]]
    plt.figure(filename + str(num))
    plt.imshow(img__)
    plt.savefig("./Results/" + str(num) + "_" + filename  )
    plt.close("all")

# the name of the file is passed as an argument of ImageAnalysis
# The number of clusters to be formed are also passed as an argument when needed
# Threshold number of pixels are also passed as an argument when needed
def ImageAnalysis(filename,clusters=3,threshold=50):
    img3 = mpimg.imread(filename)
    [x,y,z] = img3.shape
# reshaping the image from 3 dimensions to 2 dimensions, since the KMeans only
# works for 2-D matrices
    img2 = img3.reshape((x*y,z)).astype('double')
# performing KMeans with three clusters
    kmeans = KMeans(n_clusters = clusters,random_state = 0).fit(img2)
    centroid = kmeans.cluster_centers_
# reshaping the labels according to pre-clustering dimensions
    labels = kmeans.labels_
    labels = labels.reshape(x,y)
    

    distance = 45
#finding the darkest centroid index
    ind = sum(np.transpose(centroid)[:,:]).argmin()
    
# Approximately getting the starting and ending indices along x and y axis for the 
# region containing the mole
    start_y,stop_y = getIndices(labels,ind,threshold,distance,y/2)
    start_x,stop_x = getIndices(np.transpose(labels),ind,threshold,distance,x/2)
    
    
    max_x = 0
    min_x = x
    max_y = 0
    min_y = y
# finding the minimum and maximum index number for the darkest pixel for both the axes
    for i in np.arange(start_x,stop_x):
        for j in np.arange(start_y,stop_y):
            if (labels[i,j] == ind):
                if ( i < min_x ):
                    min_x = i
                if ( i > max_x ):
                    max_x = i
                if ( j < min_y ):
                    min_y = j
                if ( j > max_y ):
                    max_y = j
# finding the median by taking the mid-point of the first and last index with dark pixel    
    centro_x = int((min_x + max_x)/2)
    centro_y = int((min_y + max_y)/2)
    
    PrintLabels(labels,centroid,filename,1)
# finding the four points which will be used to make the rectangle containing the mole   
    y_rect_min,y_rect_max = RectangleRange(labels,centro_x,ind,y/2)
    x_rect_min,x_rect_max = RectangleRange(np.transpose(labels),centro_y,ind,x/2)
    
# the rectanglular region containing the mole
    new_labels = np.copy(labels[x_rect_min:x_rect_max,y_rect_min:y_rect_max])
    
    PrintLabels(new_labels,centroid,filename,2)
    
# to find the contour, the first and last dark pixel of every row is colorized
    for i in np.arange(0,new_labels.shape[0]):
        lim = np.nonzero(new_labels[i,:] == ind)[0]
        if(lim.size != 0):
            new_labels[i,lim[0]] = len(centroid)+1
            new_labels[i,lim[-1]] = len(centroid)+1
# the first and the last dark pixel of every column is colorized to find the contour       
    for i in np.arange(0,new_labels.shape[1]):
        lim = np.nonzero(new_labels[:,i] == ind)[0]
        if (lim.size != 0):
            new_labels[lim[0],i] = len(centroid)+1
            new_labels[lim[-1],i] = len(centroid)+1
    
    plt.figure(filename + str(3))
    plt.matshow(new_labels,cmap='Blues')
    plt.savefig("./Results/"+ str(3) + "_" + filename   )
    plt.close("all")
# calculating the perimeter, summing all the newly colored contour
    con_perimeter = sum(new_labels[new_labels[:,:] == len(centroid)+1])/(len(centroid)+1)
# recoloring the area contained inside the mole to find the area
    new_labels[new_labels[:,:] == ind] = len(centroid)+2
    con_area = sum(new_labels[new_labels[:,:] == len(centroid)+2])/len(centroid)+2
# radius of the circle having the same area as the mole
    con_radius = np.sqrt(con_area / np.pi)
    circ_peri = 2 * np.pi * con_radius
    ratio = con_perimeter / (circ_peri)
# all the details regarding area, perimeter, circle perimeter and the ratio are stored in this dictionary which is then returned
    ret = {"filename" : filename ,"Perimeter" : con_perimeter,"Area" : con_area,"Circ_Peri" : circ_peri, "Ratio" : ratio}
    return ret

Details = []
# running the analysis for all the 11 low risk images
for i in np.arange(1,12):
    filename = 'low_risk_%d.jpg' % (i)
    print("Starting the image processing of %s" % (filename))
    Details.append(ImageAnalysis(filename))
    print("Image processing of %s ended" % (filename))
# the details are saved in a .csv file
keys = Details[0].keys()
f = open('low_risk_details.csv','w')
writer = csv.DictWriter(f,keys)
writer.writeheader()
writer.writerows(Details)
f.close()

Details = []
# running analysis for all the medium risk images 
for i in np.arange(1,17):
    filename = 'medium_risk_%d.jpg' % (i)
    print("Starting the image processing of %s" % (filename))
    if (i==1):
        Details.append(ImageAnalysis(filename,clusters=6,threshold=30))
    else:
        Details.append(ImageAnalysis(filename))
    print("Image processing of %s ended" % (filename))
keys = Details[0].keys()
f = open('medium_risk_details.csv','w')
writer = csv.DictWriter(f,keys)
writer.writeheader()
writer.writerows(Details)
f.close()

Details = []
# running analysis for all the melanoma images 
for i in np.arange(1,28):
    filename = 'melanoma_%d.jpg' % (i)
    print("Starting the image processing of %s" % (filename))
    if(i==27):
        Details.append(ImageAnalysis(filename,clusters=6))
    else:        
        Details.append(ImageAnalysis(filename))
    print("Image processing of %s ended" % (filename))
keys = Details[0].keys()
f = open('melanoma_details.csv','w')
writer = csv.DictWriter(f,keys)
writer.writeheader()
writer.writerows(Details)

f.close()