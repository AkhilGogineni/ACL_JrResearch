#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:49:12 2022

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
    data = np.loadtxt(raw_data, dtype='str', usecols = (1,2,3,4,5,6,7,8), skiprows = 2, delimiter=",")
    
    x = data[:,range(0,7)]
    y = data[:,7]
    
    y[y == "Yes"] = "1"
    y[y == "No"] = "0"    
    
    y = y.astype(np.float64)

    return x, y

def activation(X, W):
    W = W.reshape(len(x[0]), 1)
    z = -1*np.dot(X, W)
    sig = 1/(1+math.e**(z))

    return sig


def calcCost(X, W, Y):
    costI = Y*np.log(activation(X, W)) + (1-Y) * np.log(1-activation(X, W))
    costI = -np.mean(costI)
    return costI


def calcGradient(x, weights, y):
    predicted = activation(x, weights)
    diff = predicted-y

    grad = np.dot(diff.T, x)
    #print(grad/x.shape[0])
    return grad/x.shape[0]


def dataClean(data):
    
    data[:,3][data[:,3] == "Yes"] = "1"
    data[:,3][data[:,3] == "No"] = "0"
    
    data[:,4][data[:,4] == "Yes"] = "1"
    data[:,4][data[:,4] == "No"] = "0"
    
    data[:,5][data[:,5] == "Month-to-month"] = "0"
    data[:,5][data[:,5] == "One year"] = "1"
    data[:,5][data[:,5] == "Two year"] = "2"
    
    data[:,6][data[:,6] == "Mailed check"] = "3"
    data[:,6][data[:,6] == "Bank transfer (automatic)"] = "2"
    data[:,6][data[:,6] == "Credit card (automatic)"] = "1"
    data[:,6][data[:,6] == "Electronic check"] = "0"
    
    data = data.astype(np.float64)
    
    '''
    pClassColms2 = data[:, 5:6]
    ohe2 = OneHotEncoder(categories='auto')
    passClassColms2 = ohe2.fit_transform(pClassColms2).toarray().astype(np.float64)
    '''
    
    pClassColms = data[:, 6:7]
    ohe = OneHotEncoder(categories='auto')
    passClassColms = ohe.fit_transform(pClassColms).toarray().astype(np.float64)
    
    #data = np.hstack((data[:, 0:4], passClassColms2))
    data = np.hstack((data[:, 0:5], passClassColms))


    
    data = data.astype(np.float64)
    
    
    avg = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    
    data = (data-avg)/(std)
    
    
    
    ''' normalize
    avg = data.mean(axis = 0)
    minval = np.min(data, axis = 0)
    maxval = np.max(data, axis = 0)
    
    data = (data-avg)/(maxval-minval)
    '''
    
    
    return data



x, y = readDataFile("churn_Train.csv")

x = x.reshape((x.shape[0], x.shape[1]))
y = y.reshape((y.shape[0], 1))



x = dataClean(x)

x = np.append(np.ones((x.shape[0], 1)), x, axis=1)

w = np.zeros((len(x[0]), len(y[0])))


#w = np.array([0 for i in range(x.shape[1] + 1)]).reshape(x.shape[1]+1, 1)




lr = 0.1
minVect = 0.0001
maxIterations = 10000
costArray = []

vectDiff = 1
iters = 0


while (iters < maxIterations) and (vectDiff > minVect):
    cost = calcCost(x, w, y)
    costArray.append(cost)
    
    
    gradients = calcGradient(x, w, y)
    
    #print(w - lr*(gradients.reshape(len(x[0]), 1)))
    w = w - lr*(gradients.reshape(len(x[0]), 1))
    vectDiff = np.linalg.norm(gradients)
    
    iters = iters+1
    
    
def predictTestData(xs, ws):
    return activation(xs,ws)




predictions = predictTestData(x, w)
predictions[:][predictions[:] < 0.5] = 0
predictions[:][predictions[:] >= 0.5] = 1

confused = confusion_matrix(y, predictions)


actSurvived = np.sum(y, axis = 0)/len(y) * 100

predSurvived = np.sum(predictions, axis = 0)/len(predictions) *100

percision = confused[1,1]/(confused[1,1]+confused[0,1])

recall = confused[1,1]/(confused[1,1]+confused[1,0])

f1 = 2 * ((percision*recall)/(percision+recall))

accuracy = ((confused[0,0] + confused[1,1])/ len(y))

err = ((confused[0,1] + confused[1,0])/ len(y))



print("-------------------------------------------------------")
print("final weights: ", w.reshape(1,len(x[0])))
print("-------------------------------------------------------")

print("Percentage churned in the train data: %4.4f" % actSurvived + "%")
print("Predicted percentage churned in the test data: %4.2f" % predSurvived + "%")


print("")
print("Confusion matrix - who got Churned?")
print("----------  ------------  -------------")
print("n = " + str(len(y)) + "     Predicted No  Predicted Yes")
print("Actual No"  + "   "+str(confused[0,0]) +  "           " + str(confused[0,1]))
print("Actual Yes"  + "  "+str(confused[1,0]) +  "            " + str(confused[1,1]))
print("----------  ------------  -------------")
print("-----------  ---------") 
print("Accuracy:    %5.5f" % accuracy)
print("Error Rate:  %5.5f" % err)
print("Percision:   %5.5f" % percision)
print("Recall:      %5.5f" % recall)
print("F1:          %5.5f" % f1)
print("-----------  ---------") 

print("Correct: " + str(confused[0,0] + confused[1,1])) 
print("Incorrect: " + str(confused[0,1] + confused[1,0])) 
