# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:22:02 2017

@author: Rehan Rajput
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### reading the data from the CSV file
x = pd.read_csv("parkinsons_updrs.data")    

Nt,F = x.shape

#### Making sure that test_time is an integer in the interval [0-180] for all the rows
x.test_time = x.test_time - x.test_time.min()
x.test_time = x.test_time.round()

### the value of all the features is stored as the mean of all the values having
### for each subject# and test_time
xx = x.groupby(["subject#","test_time"],as_index=False).mean()

                
### dividing the data in test and training set
data_train = xx[xx['subject#'] < 37]
data_test =  xx[xx['subject#'] > 36]

                   
### normalizing the data 
m = data_train.mean()
s = data_train.std()
data_train_norm = (data_train - m)/s
data_test_norm = (data_test - m)/s

### A function which receives a list 'FO' containing the list of unnecessary features except
### the last element of the list which is the feature to be separated (i.e: Total UPDRS)
### and put in another vector y. This function helps in getting x_train, y_train, x_test and
### y_test
def genVectors(FO,data_train_norm):
    x_train = data_train_norm
    for i in FO:
        y_train = data_train_norm[i]
        x_train = x_train.drop(i,1)
    return x_train,y_train


x_train,y_train = genVectors(['subject#','test_time','age','sex','total_UPDRS'],data_train_norm)
x_test,y_test = genVectors(['subject#','test_time','age','sex','total_UPDRS'],data_test_norm)


N = len(x_train)
### This function applies the MSE method on a given set of data (i.e: x and y)
def MSE(x_train,y_train):
    a = np.linalg.pinv(x_train.T.dot(x_train))
    b = a.dot(x_train.T.dot(y_train))
    return b

### This function plots the results calculated against the true values
def plot(y_test,y_result,title):
    area = np.pi*3
    plt.scatter(y_test,y_result,c = ('red','blue'),s=area)
    plt.title(title)
    plt.xlabel('True result (red)')
    plt.ylabel('calculated result (blue)')
    plt.grid()
    plt.show()
    plt.hist(y_test-y_result,bins=50,color=('green'))
    plt.title(title)
    plt.grid()
    plt.show()
### This function plots the value of w for each method
def plot_w(w,title):
    plt.plot(w)
    plt.title(title)
    plt.xlabel("index number")
    plt.ylabel("value of 'w'")
    plt.show()
### All the regrssion methods are trained on training set and then applied on training 
### and test sets. Later on, the error is calculated on the calculated 'y' for both 
### the training and the test set, and the error is calculated.
### Scatter plots and histograms are generated for the calculated and true value of 'y'
### for each method. Moreover, the value of 'w' for each method is also plotted
### Most of the variables are named so that they're self-explanatory

### Most of the arguments to the regression methods are named such that they're
### self-explanatory

### MSE trained using training data and calculating vector 'w'
w_MSE = MSE(x_train,y_train)
y_MSE = x_test.dot(w_MSE)
y_MSE_train = x_train.dot(w_MSE)

plot(y_train,y_MSE_train,'MSE Train')
plot(y_test,y_MSE,'MSE Test')
plot_w(w_MSE,"W vector for MSE")

### This function calculates the error between the calculated value of 'y' and true value
### of 'y'
def minError(y_cal,y_test):
    sum = np.linalg.norm(y_cal.values-y_test.values)
    return (sum)/y_test.size

minError_MSE = minError(y_MSE,y_test)
minError_MSE_train = minError(y_MSE_train,y_train)

### This function applies the gradient algorithm regression, it creates a list of
### dictionaries ('vals'), with each dictionary containing the vector 'w',
### the corresponding 'error' and the corresponding 'gamma' value. The gamma values
### range from 10^-7 to 10^8 in the list. Later on, the function selects the dictionary
### (having the value of 'w', 'error' and 'gamma') with the lowest amount of error, 
### in order to make sure that the value of gamma is optimal
def GradientAlgorithm(x_train,y_train,x_test,y_test,rand_vector,max_iter,epsilon):
    vals = []
    for n in range(-7,8):
        gamma = pow(10,n)
        w = rand_vector
        for i in np.arange(max_iter):
            a = -2*x_train.T.dot(y_train)
            b = 2*x_train.T.dot(x_train.dot(w))
            grad = a+b
            w1 = w - gamma*grad
            if (np.linalg.norm(w1.values - w) < epsilon):
                w =  w1.values
                break
            w=w1.values
        err = minError(x_test.dot(w),y_test)
        vals.append({"gamma":gamma,"w":w,"error":err})
    ret = sorted(vals,key=lambda x : float(x.get('error')))
    return ret[0]

### random seed is zero, so that we get the same random vector everytime we run the
### code. Comment the line below to get 'a real' random vector everytime
np.random.seed(0)
rand_vector = np.random.rand(len(x_train.columns))
GradAlgo_Details = GradientAlgorithm(x_train,y_train,x_test,y_test,rand_vector,300,pow(10,-3))
gamma_GradAlgo = GradAlgo_Details.get('gamma')
w_GradAlgo = GradAlgo_Details.get('w')
y_GradAlgo = x_test.dot(w_GradAlgo)
y_GradAlgo_train = x_train.dot(w_GradAlgo)


minError_GradAlgo = minError(y_GradAlgo,y_test)
minError_GradAlgo_train = minError(y_GradAlgo_train,y_train)
plot(y_train,y_GradAlgo_train,'GradAlgo Train')
plot(y_test,y_GradAlgo,'GradAlgo Test')
plot_w(w_GradAlgo,"W vector for GradAlgo")
### this function finds the value of 'w' using the steep descent regression method
def SteepDescent(x_train,y_train,rand_vector,max_iter,epsilon):
    w = rand_vector
    H = 4*x_train.T.dot(x_train)
    for i in range(max_iter):
        grad= -2*x_train.T.dot(y_train) + 2*x_train.T.dot(x_train.dot(w))
        num= (np.linalg.norm(grad.values))**2
        den= grad.T.dot(H.dot(grad))
        gamma = num/den
        w1 = w - gamma*grad
        if (np.linalg.norm(w1.values - w) < epsilon):
            return w1.values
        w=w1.values
    return w

w_SteepDescent = SteepDescent(x_train,y_train,rand_vector,100,pow(10,-3))
y_SteepDescent = x_test.dot(w_SteepDescent)
y_SteepDescent_train = x_train.dot(w_SteepDescent)
plot(y_train,y_SteepDescent_train,'SteepDescent Train')
plot(y_test,y_SteepDescent,'SteepDescent Test')
plot_w(w_SteepDescent,"W vector for SteepDescent")

minError_SteepDescent_train = minError(y_SteepDescent_train,y_train)
minError_SteepDescent = minError(y_SteepDescent,y_test)


### This function applies the ridge regression method to find the vector 'w'
def Ridge(x_train,y_train,lmbda):
    iden = lmbda*np.identity(x_train.columns.size)
    a = np.linalg.pinv(x_train.T.dot(x_train) + iden)
    b = a.dot(x_train.T.dot(y_train))
    return b
### The value of lambda was manually set to be 60 after manually testing different 
### values and checking the error rate
w_Ridge = Ridge(x_train,y_train,60)
y_Ridge = x_test.dot(w_Ridge)
y_Ridge_train = x_train.dot(w_Ridge)
plot(y_train,y_Ridge_train,'Ridge Train')
plot(y_test,y_Ridge,'Ridge Test')
plot_w(w_Ridge,"W vector for Ridge")
minError_Ridge_train = minError(y_Ridge_train,y_train)
minError_Ridge= minError(y_Ridge,y_test)

### calculates the 'L' number of features whose eigenvalues have a sum 
### equal to 95% of the sum of all the eigenvalues of all the features
def L_features(A):
    P = sum(A)
    A = A.sort_values(ascending = False)
    for i in range(3,len(A)+1):
        temp = A.head(i)
        L = sum(temp)
        if L >= (0.95)*P:
            return temp
    return temp


### Extracting only the eigenvectors corresponding to the 'L' features found using the above
### method
def minimize_U(U,A):
    min_set =  set(U.columns) - set(A.index)
    for i in min_set:
        U = U.drop(i,1)
    return U

def PCA(x_train):
    ### PCA Method 
    N = len(x_train)
    ### Co-variance matrix 
    Rx_train = (1/N)*x_train.T.dot(x_train)
    ### to diagonalize it, we find a matrix of eigenvectors 'U' and a diagonal matrix
    ### of eigen values 'A'
    A,U = np.linalg.eig(Rx_train)
    ### converting these matrices to panda dataseries and dataframe to manipulate them
    ### with ease
    A = pd.Series(A,index = Rx_train.columns)
    U = pd.DataFrame(U,index = Rx_train.index,columns = Rx_train.columns)
    
    A = L_features(A)
    U = minimize_U(U,A)
    A = np.diag(A)
    A = np.linalg.pinv(A)
    
    return U,A
### calculating the inverse of the diagonal matrix of L eigenvalues beforehand, to make
### the calculation of 'w' easier


[U,A] = PCA(x_train)
### calculating the 'w' vector using PCR method
w_PCR = U.dot(A.dot(U.T.dot(x_train.T.dot(y_train))))
y_PCR = (1/len(x_test)) * x_test.dot(w_PCR)
y_PCR_train = (1/N) * x_train.dot(w_PCR)

minError_PCR_train = minError(y_PCR_train,y_train)
plot(y_train,y_PCR_train,"PCR Train")

minError_PCR = minError(y_PCR,y_test)
plot(y_test,y_PCR,"PCR Test")
plot_w(w_PCR.values,"W vector for PCR")

########## Cross Validation ##############
cross_x = []    #### a list of 5 subsets of training set
cross_y = []    #### a list of 5 subsets of classes of training sets
for i in np.arange(0,5):
    l = N//5    ### length of each subset
    
    temp_xtrain = x_train[x_train.index >= i*l]
    temp_xtrain = temp_xtrain[temp_xtrain.index < (i+1)*l]
    cross_x.append(temp_xtrain)
    
    temp_ytrain = y_train[y_train.index >= i*l]
    temp_ytrain = temp_ytrain[temp_ytrain.index < (i+1)*l]
    cross_y.append(temp_ytrain)

#### a list of errors for each K-fold training
cross_MSE_err = []
cross_Grad_err = []
cross_Steep_err = []
cross_Ridge_err = []
cross_PCR_err = []

for i in np.arange(0,5):
    ## temp_xtrain is composed of 4 subsets of training set
    temp_xtrain = pd.concat([cross_x[j] for j in np.arange(0,5) if j!=i])
    ## temp_xtest contains the validation subset
    temp_xtest = cross_x[i]
    
    ## temp_ytrain is composed of 4 class labels subsets of training set
    temp_ytrain = pd.concat([cross_y[j] for j in np.arange(0,5) if j!=i])
    ## temp_ytest contains the class labels of validation subset
    temp_ytest = cross_y[i]
    
    ## MSE for cross validation
    temp_w = MSE(temp_xtrain,temp_ytrain)
    temp_y = temp_xtest.dot(temp_w)
    cross_MSE_err.append(minError(temp_ytest,temp_y))
    
    ## GradAlgo for cross validation
    temp_rand = np.random.rand(len(temp_xtrain.columns))
    temp_grad_details = GradientAlgorithm(temp_xtrain,temp_ytrain,temp_xtest,temp_ytest,temp_rand,300,pow(10,-3))
    temp_gamma = temp_grad_details.get('gamma')
    temp_w = temp_grad_details.get('w')
    temp_y = temp_xtest.dot(temp_w)
    cross_Grad_err.append({'error': minError(temp_y,temp_ytest) , 'gamma':temp_gamma})
    
    ## SteepDescent for cross validation
    temp_w = SteepDescent(temp_xtrain,temp_ytrain,temp_rand,100,pow(10,-3))
    temp_y = temp_xtest.dot(temp_w)
    cross_Steep_err.append(minError(temp_ytest,temp_y))
    
    ## Ridge for cross-validation
    temp_w = Ridge(temp_xtrain,temp_ytrain,60)
    temp_y = temp_xtest.dot(temp_w)
    cross_Ridge_err.append(minError(temp_ytest,temp_y))
    
    ## PCA for cross validation
    [temp_U,temp_A] = PCA(temp_xtrain)
    temp_w = temp_U.dot(temp_A.dot(temp_U.T.dot(temp_xtrain.T.dot(temp_ytrain))))
    temp_y = (1/len(temp_xtest)) * temp_xtest.dot(temp_w)
    cross_PCR_err.append(minError(temp_ytest,temp_y))