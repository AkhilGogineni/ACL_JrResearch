#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:24:48 2022

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
    data = np.loadtxt(raw_data, usecols = (2,3,4,5,6,7,8,9,10,14), skiprows = 1, delimiter=",")
    
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




x,y = readDataFile("Cereal_Train.csv")


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


LR = 0.007
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


#plot data
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(x[:, 1], y, 'rx')
ax1.set(title = 'Data with Best Fit Line', xlabel = 'X values', ylabel = 'Y values')

# todo: plot the best fit line
'''
  - use np.argmin() to find the index of the minimum value in x
  - use np.argmax() to find the index of the maximum value in x
  - create an array with the min and max x values
  - create an array with the predicted y values: 2 x values * weights 
'''

xMin = np.argmin(x[:,1])
xMax = np.argmax(x[:,1])
xVals = [x[xMin,1], x[xMax,1]]
predictedY = [xVals[0]*weights[1, :], xVals[1]*weights[1, :]]

ax1.plot(xVals, predictedY+weights[0], linewidth=1, linestyle='-', color='Red')


# plot cost
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(iterations), costArray, color='blue')
ax2.set(title = 'Cost vs. Iterations', xlabel = 'iterations', ylabel = 'cost')


#test data
x1,y1 = readDataFile("Cereal_Test.csv")


x1 = (x1 - avg)/std



ones1 = np.ones((len(x1), 1))
x1 = np.hstack((ones1,x1))


y1 = y1.reshape((len(y1),1))
rows1 = x1.shape[1]
cols1 = 1



prediction = np.dot(x1, weights)
diff = prediction - y1

    
    