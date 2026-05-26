#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:14:54 2022

@author: akhilgogineni
"""

import numpy as np
import matplotlib.pyplot as plt


def readDataFile():
    fileName = 'murdersunemployment.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    #loadtxt defaults to float
    data = np.loadtxt(raw_data, usecols = (2,3,4), skiprows = 1, delimiter=",")
    
    x = data[:,0:2]
    y = data[:, 2]
   


    
    return x, y





def costFunc(x, weights, y):
    # todo: compute the cost using numpy, not for-loops
    predvals = np.dot(x, weights)
    
    sqerr = np.square(predvals - y)
    merror = np.mean(sqerr, axis = 0)
    return merror
    

        
def gradDesc(x, weights, y):
    predvals = np.dot(x.reshape(20,3),weights.reshape(3,1))
    
    err = predvals-y.reshape(20,1)
    
    #grad = np.mean((err*x),axis = 1)
    grad = np.dot(err.T,x)      
    return grad




x,y = readDataFile()

avg = np.mean(x, axis = 0)
#print("avg: " + str(avg))
std = np.std(x, axis = 0)
#print("std: " + str(std))

x = (x - avg)/std

ones = np.ones((len(x), 1))
x = np.hstack((ones,x))


y = y.reshape((20,1))
rows = x.shape[1]
cols = 1


weights = np.zeros((rows, cols)).reshape(3,1)


 



LR = 0.003
costArray = []
gradArray = []
weightArray = []

iterations = 0
maxIter = 50000
minDiff = .01
vectDiff = 1

print("inital weights: " + str(weights.reshape(1,3)))
while iterations < maxIter and vectDiff > minDiff:
    
    
    cost = costFunc(x, weights, y)
    costArray.append(cost)
        
    gradients = gradDesc(x, weights, y)
    gradArray.append(gradients)
    
    
    weights = weights - (LR*(gradients.reshape(3,1)))
    weightArray.append(weights)
    
    vectDiff = np.linalg.norm(gradients)
    iterations +=1

    

weights = weights.reshape(1,3)
print("final weights: " + str(weights.reshape(3,1)))
print(iterations)

predX = np.array([22.4, 8.6]).reshape(1,2)
predX = (predX-avg)/std
predX = np.append(np.ones((1,1)), predX, axis =1)
pred = np.dot(predX, weights)


    
    