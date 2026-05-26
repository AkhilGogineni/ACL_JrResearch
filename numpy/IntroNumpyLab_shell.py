# -*- coding: utf-8 -*-
"""
Google numpy to figure out the methods you need to implement the tasks within the comments
Make sure you look at Spyder's Variable explorer to validate your results.

Set a breakpoint at line 14 (where the matrix X is created) and step through the program.
"""

import numpy as np
import numpy.random as random 

#1 create a 20x2 array with random values
X = random.rand(20,2)

#2 set a variable, row, to the number of rows
row = len(X)

#3 multipy the first column by 20 - multipy by a scalar
X[:,0] *= 20


#3 multipy the 2nd column by 1000
X[:, 1] *= 1000


#4 calculate the minimum of column 0
minimum = X[:,0].min()


#4 calculate the max of column 1
maximum = X[:, 0].max()

#5 print the max and min:    "min of col 0: xyz,  max of col 1: abc"
print('min of col 0: ' + str(minimum) + ', max of col 1: '  + str(maximum))

#6 calculate the average of the 1st column
avg = X[:, 0].mean()

#6 calculate the average of both columns  => array of 2 elements
avgBoth = X.mean()

#7 determine the number of rows and columns in the matrix X
print("rows, cols: " + str(X.shape))
rows_cols = X.shape


#8 create a (rows x 1) np array of all zeros using np.zeros
arr = np.zeros((row, 1))

#9 add that np array of all zeros as a third column to X using np.hstack() -- make sure you specify a tuple
X = np.hstack((X, arr))


#10 add column 0 and 1 of X into column 2  
X[:, 2] = np.add(X[:, 0], X[:,1])


#11 slicing: store a section of rows or columns into a numpy darray
#   store rows 3, 4, and 5 into sliceRowsX
sliceRowsX = X[3:6]

#12 store columns 0 and 2 into sliceColsX
sliceColsX = X[:,[0,2]]



