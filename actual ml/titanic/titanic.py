#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:33:52 2022

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
    data = np.loadtxt(raw_data, dtype='str', usecols = (1,2,5,6,7,8,10), skiprows = 1, delimiter=",")
    
    x = data[:,range(1,7)]
    y = data[:,0]
    
    
    y = y.astype(np.float64)
    
    return x, y

def activation(X, W):
    W = W.reshape(len(x[0]), 1)
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

    grad = np.dot(diff.T, x)
    return grad/x.shape[0]

def dataClean(data):
    data[:,1][data[:,1] == "male"] = "0"
    data[:,1][data[:,1] == "female"] = "1"
    

    data[:,2][data[:,2]==""] = np.nan
    
    
    data[:,5][data[:,5]==""] = np.nan


    data = data.astype(np.float64)



    saved_mean = np.nanmean(data[:, 2])
       
    data[:,2][np.isnan(data[:,2])]=saved_mean
    
    
    saved_mean2 = np.nanmean(data[:, 5])
       
    data[:,5][np.isnan(data[:,5])]=saved_mean2


    avg = data[:,2].mean(axis = 0)
    minval = np.min(data[:,2], axis = 0)
    maxval = np.max(data[:,2], axis = 0)
    
    data[:,2] = (data[:,2]-avg)/(maxval-minval)
    
    avg2 = data[:,5].mean(axis = 0)
    minval2 = np.min(data[:,5], axis = 0)
    maxval2 = np.max(data[:,5], axis = 0)
    
    data[:,5] = (data[:,5]-avg2)/(maxval2-minval2)

    
    pClassColms = data[:, 0:1]
    ohe = OneHotEncoder(categories='auto')
    passClassColms = ohe.fit_transform(pClassColms).toarray().astype(np.float64)
    
    ndata = np.hstack((passClassColms, data[:, 1:]))
    
    
    #print("X after normalization: \n", data)

    return ndata



x, y = readDataFile("TitanicSurvival_Train.csv")

x = x.reshape((x.shape[0], x.shape[1]))
y = y.reshape((y.shape[0], 1))

x = dataClean(x)
w = np.array([0 for i in range(x.shape[1] + 1)]).reshape(x.shape[1]+1, 1)



x = np.append(np.ones((x.shape[0], 1)), x, axis=1)



lr = 0.2
minVect = 0.0001
maxIterations = 10000
costArray = []

vectDiff = 1
iters = 0

print("inital weights: ", w.reshape(1,9))

while (iters < maxIterations) and (vectDiff > minVect):
    cost = calcCost(x, w, y)
    costArray.append(cost)
    
    
    gradients = calcGradient(x, w, y)
    
    
    w = w - lr*(gradients.reshape(len(x[0]), 1))
    vectDiff = np.linalg.norm(gradients)
    
    iters = iters+1



# plot cost
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(iters), costArray, color='blue')
ax2.set(title = 'Cost vs. Iterations', xlabel = 'iterations', ylabel = 'cost')

def testData(fileName):
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    #loadtxt defaults to float
    data = np.loadtxt(raw_data, dtype='str', usecols = (1,4,5,6,7,9), skiprows = 1, delimiter=",")
    
    return data

def predictTestData(xs, ws):
    return activation(xs,ws)

testingData = testData("TitanicSurvival_Test.csv")
testingData = dataClean(testingData)
testingData = np.append(np.ones((testingData.shape[0], 1)), testingData, axis=1)


predictions2 = predictTestData(testingData, w)
predictions2[:][predictions2[:] < 0.5] = 0
predictions2[:][predictions2[:] >= 0.5] = 1




predictions = predictTestData(x, w)
predictions[:][predictions[:] < 0.5] = 0
predictions[:][predictions[:] >= 0.5] = 1

confused = confusion_matrix(y, predictions)


actSurvived = np.sum(y, axis = 0)/len(y) * 100

predSurvived = np.sum(predictions2, axis = 0)/len(predictions2) *100

percision = confused[1,1]/(confused[1,1]+confused[0,1])

recall = confused[1,1]/(confused[1,1]+confused[1,0])

f1 = 2 * ((percision*recall)/(percision+recall))

accuracy = ((confused[0,0] + confused[1,1])/ len(y))

err = ((confused[0,1] + confused[1,0])/ len(y))


print("-------------------------------------------------------")
print("final weights: ", w.reshape(1,9))
print("-------------------------------------------------------")

print("Percentage survived in the train data: %4.4f" % actSurvived + "%")
print("Predicted percentage survived in the test data: %4.2f" % predSurvived + "%")


print("")
print("Confusion matrix - who survived?")
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



#np.savetxt("titanicpredictions.csv", predictions2, delimiter = ",")

np.savetxt('titanicprediction2.csv', predictions2, fmt="%d", delimiter=",")
