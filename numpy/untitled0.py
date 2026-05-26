#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:26:12 2022

@author: akhilgogineni
"""
import numpy as np
from scipy import stats as st
import numpy.linalg as lg
import csv


def readDataFile(fileName):
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    # loadtxt defaults to floats, use dtype to specify string
    # usecols chooses the columns to use, by default, all columns are used.
    # skiprows skips a header row if you need to
    data = np.loadtxt(raw_data, delimiter=",", dtype = 'str')
    stats = data[:14,:].astype(float)
    
    return stats


psych = readDataFile('psych1.csv')

mean = psych.mean(axis = 0)
median = np.median(psych, axis = 0)
mode = st.mode(psych, axis = 0)
rangex = np.ptp(psych, axis = 0)