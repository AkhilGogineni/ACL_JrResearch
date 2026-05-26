#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:11:18 2022

@author: akhilgogineni
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix


def readDataFile(fileName):
    raw_data = open(fileName, 'rt')
    #loadtxt defaults to float
    data = np.loadtxt(raw_data, dtype='str', usecols = (0,1,2,3,4), skiprows = 1, delimiter=",")
    
    x = data[:,range(0,4)]
    y = data[:,4]
    
    
    y = y.astype(np.float64)
    
    return x, y

def activation(X, W):
    W = W.reshape(len(x[0]), len(y[0]))
    z = -1*np.dot(X, W)
    sig = 1/(1+math.e**(z))

    return sig


def calcCost(X, W, Y):
    costI = Y*np.log(activation(X, W)) + (1-Y) * \
        np.log(1-activation(X, W))
    costI = -np.mean(costI)
    return costI


def calcGradient(x, weights, y):
    predicted = activation(x, weights)
    diff = predicted-y

    grad = np.dot(x.T, diff)
    return grad/x.shape[0]


def dataClean(data):
    
    data = data.astype(np.float64)

    avg = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    
    data = (data-avg)/(std)
    
    ''' standardize
    avg = data.mean(axis = 0)
    minval = np.min(data, axis = 0)
    maxval = np.max(data, axis = 0)
    
    data = (data-avg)/(maxval-minval)
    '''
    
    
    return data



x, y = readDataFile("irisdata.csv")

x = x.reshape((x.shape[0], x.shape[1]))
y = y.reshape((y.shape[0], 1))

pClassColms = y
ohe = OneHotEncoder(categories='auto')
passClassColms = ohe.fit_transform(pClassColms).toarray().astype(np.float64)

yorig = y
y = passClassColms

x = dataClean(x)

x = np.append(np.ones((x.shape[0], 1)), x, axis=1)

w = np.zeros((len(x[0]), len(y[0])))


#w = np.array([0 for i in range(x.shape[1] + 1)]).reshape(x.shape[1]+1, 1)




lr = 0.2
minVect = 0.0001
maxIterations = 10000
costArray = []

vectDiff = 1
iters = 0


while (iters < maxIterations) and (vectDiff > minVect):
    cost = calcCost(x, w, y)
    costArray.append(cost)
    
    
    gradients = calcGradient(x, w, y)
    
    
    w = w - lr*(gradients.reshape(len(x[0]), len(y[0])))
    vectDiff = np.linalg.norm(gradients)
    
    iters = iters+1
    
    
    
def predictTestData(xs, ws):
    return activation(xs,ws)


predictions = predictTestData(x, w)

predColumn = np.argmax(predictions,axis=1)

predColumn = predColumn + 1

predColumn.reshape(150, 1)

correct = yorig.reshape(150,1) - predColumn.reshape(150,1)

correctCount = len(correct) - np.count_nonzero(correct)

print("Number Correct: " + str(correctCount))
print("Number Incorrect: " + str(np.count_nonzero(correct)))


    

