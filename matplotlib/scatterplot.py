#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:17:06 2022

@author: akhilgogineni
"""

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)

data1 = np.zeros((50,2))


y_intercept = 5
slope = 2

delta = 10
for i in range(0, 50):
    data1[i,0] = (int(i))
    data1[i,1] = (int(random.randint(slope*i-delta, slope*i+delta)) + y_intercept)
    
fig = plt.figure()

ax1 = fig.add_axes([0.1,0.15,0.8,0.7])# [left, bottom, width, height]

ax1.set_title('Scatter Plot - linear regression')
ax1.set_xlabel('X - Independent Variable')
ax1.set_ylabel('Y - Dependent Variable')
ax1.scatter(data1[:, 0], data1[:, 1], marker='x', label = "data1")


data2 = np.zeros((50,2))


y_intercept1 = 5
slope1 = -1

delta = 10
for i in range(0, 50):
    data2[i,0] = (int(i))
    data2[i,1] = (int(random.randint(slope1*i-delta, slope1*i+delta)) + y_intercept1)
    



ax1.scatter(data2[:, 0], data2[:, 1], marker='o', label = "data2", color ="red")


ax1.legend(loc="upper left")







