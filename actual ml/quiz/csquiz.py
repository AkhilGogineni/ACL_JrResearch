#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:24:55 2022

@author: akhilgogineni
"""

import numpy as np
import matplotlib.pyplot as plt


def readDataFile(fileName):
    #fileName = 'Cricket Chirp Data.csv'
    #fileName = "CricketChirpData_Train.csv"
    
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    #loadtxt defaults to float
    data = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6,7,8,9), skiprows = 20, delimiter=",")
    
    x = data[:,0:9]
    y = data[:,9]
   


    
    return x, y




def costFunc(x, weights, y):
    # todo: compute the cost using numpy, not for-loops
    predvals = np.dot(x, weights)
    
    sqerr = np.square(predvals - y)
    merror = np.mean(sqerr, axis = 0)
    return merror
    

        
def gradDesc(x, weights, y):
    predvals = np.dot(x.reshape(len(x),len(weights)),weights.reshape(len(weights),1))
    
    err = predvals-y.reshape(len(x),1)
    
    #grad = np.mean((err*x),axis = 1)
    grad = np.dot(err.T,x)      
    return grad




x,y = readDataFile("2B_Train.csv")


avg = np.mean(x, axis = 0)
#print("avg: " + str(avg))
std = np.std(x, axis = 0)
#print("std: " + str(std))

avgy = np.mean(y, axis = 0)
#print("avg: " + str(avg))
stdy = np.std(y, axis = 0)
#print("std: " + str(std))



x = (x - avg)/std




ones = np.ones((len(x), 1))
x = np.hstack((ones,x))


y = y.reshape((len(y),1))
rows = x.shape[1]
cols = 1


weights = np.zeros((rows, cols))
weights = weights.reshape(len(weights),1)

print("weights" + str(len(weights)))


LR = 0.01
costArray = []
gradArray = []
weightArray = []

iterations = 0
maxIter = 50000
minDiff = .01
vectDiff = 1
print("inital weights: " + str(weights.reshape(1,len(weights))))

while iterations < maxIter and vectDiff > minDiff:
    
    
    cost = costFunc(x, weights, y)
    costArray.append(cost)
        
    gradients = gradDesc(x, weights, y)
    gradArray.append(gradients)
    
    
    weights = weights - (LR*(gradients.reshape(len(weights),1)))
    weightArray.append(weights)
    
    vectDiff = np.linalg.norm(gradients)
    iterations +=1

    

print("final weights: " + str(weights.reshape(1,len(weights))))
print(iterations)



#test data
x1,y1 = readDataFile("2B_Test.csv")


x1 = (x1 - avg)/std



ones1 = np.ones((len(x1), 1))
x1 = np.hstack((ones1,x1))


y1 = y1.reshape((len(y1),1))
rows1 = x1.shape[1]
cols1 = 1



prediction = np.dot(x1, weights)
diff = prediction - y1
    
    