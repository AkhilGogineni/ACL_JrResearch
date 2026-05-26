#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:26:50 2022

@author: akhilgogineni
"""
import numpy as np
import numpy.linalg as lg
import csv
import numpy.random as random
import matplotlib.pyplot as plt

fig = plt.figure()


def readDataFile(fileName):
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    # loadtxt defaults to floats, use dtype to specify string
    # usecols chooses the columns to use, by default, all columns are used.
    # skiprows skips a header row if you need to
    data = np.loadtxt(raw_data, skiprows = 1, delimiter=",", dtype = 'str')
    abnames = data[:, 0]
    names = data[:, 3]
    lat = data[:,2].astype(float)
    long = data[:,1].astype(float)
    
    return names, lat, long, abnames


stnames, lat, long, abstnames = readDataFile('state_lat_long.csv')

ax1 = fig.add_axes([0.1,0.15,0.8,0.7])# [left, bottom, width, height]

ax1.scatter(lat, long, marker='x', label = "data2", color ="red")

ax1.set_title('Average Longitude and Latitude of US States')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')