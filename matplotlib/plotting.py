#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:29:23 2022

@author: akhilgogineni
"""

import numpy as np
from matplotlib import pyplot as plt

#generate 20 datapoints between 0 and 10
x = np.linspace(0,10, 11) 
y= x**2 
z = x**3

fig = plt.figure()

ax = fig.add_axes([0.1,0.2,0.8,0.7])
# [%left, %bottom, %width, %height]
# If you make it .5 from the left, but set width at 1.0, it will cut off part of the graph

#Plot the x2 line
# x and y are NOT points but arrays of points

ax.plot(x,y,linewidth=3, linestyle='dashed', color= '#800080')

#You plot the x3 line on your own
#Change the color and the linestyle



ax.set_title('Johns Plot')
ax.set_xlabel('Xs to be squared')
ax.set_ylabel('function values')
# Go ahead and create a title, xlabel and
#y_label for ax2
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])# [left, bottom, width, height]
ax2 = fig.add_axes([0.2,0.5,0.4,0.3])# [left, bottom, width, height]

ax1.plot(x,y,linewidth=3, linestyle='dashed', color='cornflowerblue')

#create a new figure object
#Where do you think figure ax1 is going and where is ax2 going in the window?
ax2.plot(y,x,linewidth=3, linestyle='dashed', color='cornflowerblue')

#create a new figure object

#Where do you think figure ax1 is going and where is ax2 going in the window?


#this will just add a red line to the ax1 subplot
ax1.plot([1, 5.5, 6.5, 10], [10, 25, 35, 45], color='r')
#This will plot the points
ax1.scatter(y,x)
