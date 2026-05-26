#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:45:16 2022

@author: akhilgogineni
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:41:31 2020

@author: JCHAPIN
"""
# Define the activation, calcCost and calcGradient functions
# for Logistic Regression (Classifier)

import numpy as np
import math
import matplotlib.pyplot as plt


def activation(X, W):
    W = W.reshape(3, 1)
    z = -1*np.dot(x, W)
    
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

x = np.array([[1., 5., 10.],
              [1., 5., 6.],
              [1., -4., 0.],
              [1., -5., -6.],
              [1., 4., 8.],
              [1., 0., 0.],
              [1., -6., 4.],
              [1., 4., 0.],
              [1., 2., -5.],
              [1., 4., -5.],
              [1., -4., -3.]]).astype(float)

y = np.array([[1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.]]).astype(float)
y = y.reshape(11, 1)
w = np.array(np.zeros((x.shape[1], 1)))


costArray = []
gradArray = []
weightArray = []
sigArray = []

LR = 0.2
minVect = 0.0001
maxIter = 10000


vectDiff = 1
iters = 0

print("inital weights: " + str(w.reshape(1, 3)))
#print("W0", w.shape)
#w = np.expand_dims(w,axis=-1)
while iters < maxIter and vectDiff > minVect:
    

    cost = calcCost(x, w, y)
    costArray.append(cost)

    gradients = calcGradient(x, w, y)
    gradArray.append(gradients)
    #print("g", gradients.shape)
    w = w - LR*(gradients.reshape(3, 1))
    
    vectDiff = np.linalg.norm(gradients)
    iters += 1

print(activation(x, w))
print(calcCost(x, w, y))
print(calcGradient(x, w, y))


#plot data
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
#ax1.plot(x[:, 1], x[:,], 'rx')
ax1.set(title = 'Decision Boundary for Data', xlabel = 'X1', ylabel = 'X2')

# todo: plot the best fit line
'''
  - use np.argmin() to find the index of the minimum value in x
  - use np.argmax() to find the index of the maximum value in x
  - create an array with the min and max x values
  - create an array with the predicted y values: 2 x values * weights 
'''



for d, sample in enumerate(x):
    if y[d] == 0:
        plt.plot(sample[1], sample[2], "rx")
    else:
        plt.plot(sample[1], sample[2], "bx")

slope = -w[1]/w[2]
intercept = -w[0]/w[2]
xVals = [np.min(x[:,1]), np.max(x[:,1])]
predictions = xVals*slope+intercept

ax1.plot(xVals, predictions)

# plot cost
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(iters), costArray, color='blue')
ax2.set(title = 'Cost vs. Iterations', xlabel = 'iterations', ylabel = 'cost')
