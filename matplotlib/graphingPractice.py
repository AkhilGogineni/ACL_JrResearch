#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:30:35 2022

@author: akhilgogineni
"""

import numpy as np 
from matplotlib import pyplot as plt

x = np.linspace(0,10,20)
y = x**2
z = x**3

#fig = plt.figure()

#ax1 = fig.add_axes([0.1,0.2,0.8,0.7])
# [%left, %bottom, %width, %height]
# If you make it .5 from the left, but set width at 1.0, it will cut off part of the graph

#ax1.plot(x,y,linewidth=3, 
#linestyle='dashed', 
#color='peachpuff')


#ax1.plot(x,z,linewidth=3, 
#linestyle='dashed', 
#color='#703255')

#ax1.set_title('My Plot')
#ax1.set_xlabel('Xs to be squared')
#ax1.set_ylabel('function values')

#ax1.plot(x,y,linewidth=3, linestyle='dashed', color='peachpuff', label='x^2')
#ax1.plot(x,z,linewidth=3,linestyle='dashdot', color='#703255', label='x^3')
#ax1.legend()

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])# [left, bottom, width, height]
ax2 = fig.add_axes([0.2,0.5,0.4,0.3])# [left, bottom, width, height]

ax1.plot(x,y,linewidth=3, linestyle='dashed', color='cornflowerblue')
ax2.plot(y,x,linewidth = 3, linestyle='dashed', color='peachpuff')

ax1.plot([1, 5.5, 6.5, 10], [10, 25, 35, 45], color='r')

#ax1.scatter(y,x)

plt.savefig('test.png')


data = np.zeros((50,2))
y_intercept = 5
slope = 2

delta = 20
for i in range(len(data)):
    data[i,0] = i
    data[i,1] = i*slope + y_intercept + np.random.randint(-delta,delta)

ax3 = ax1 = fig.add_axes([0.1,0.1,0.8,0.8])# [left, bottom, width, height]
ax3.plot(data[0, :], data[1,:], linewidth = 3, linestyle = 'dashed', color = '#702839')