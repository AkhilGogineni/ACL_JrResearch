#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:51:07 2022

@author: akhilgogineni
"""

import numpy as np
import matplotlib.pyplot as plt

#base vals
bias = 0
slope = .5
numPoints = 4

#2d array
xyarr = np.zeros((numPoints, 2))

#populating array
xyarr[:, 0] = np.linspace(0, numPoints-1, numPoints)
xyarr[:,1] = xyarr[:,0]*slope+bias

#x and y arrays
x = xyarr[:,0]
y = xyarr[:,1]

#test slopes
numSlopes = 4
testSlopes = np.linspace(0, 4*slope, numSlopes)


def cost(bias, x, slope, y):
    predictedvals = y
    actualvals = x*slope+bias
    merror = (np.sum((actualvals - predictedvals)**2))/len(y)
    return merror



costarr = np.zeros((numSlopes, 2))
costarr[:, 0] = testSlopes

for i in range(numSlopes):
    costarr[i, 1] = cost(bias, x, testSlopes[i], y)
    
    
fig = plt.figure()
ax = fig.add_axes([0.1,0.15,0.8,0.7])# [left, bottom, width, height]



ax.plot(costarr[:,0], costarr[:,1], color ="blue")
ax.set_xlabel("Slopes")
ax.set_ylabel("Mean Squared Error")
ax.set_title('Slopes vs. Mean Squared Error')