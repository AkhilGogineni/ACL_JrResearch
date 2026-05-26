# -*- coding: utf-8 -*-
"""
Google numpy to figure out the methods you need to implement the tasks within the comments
Make sure you look at Spyder's Variable explorer to validate your results.

Name: 

"""

import numpy as np
import numpy.random as random 


# =============================================================================
# Broadcasting - How does Numpy handle the operators when the dimensions of 2 matrices do not align?
#    vertical broadcasting
#    horizontal broadcasting
#    scalar broadcasting
# =============================================================================

#set a breakpoint on the line below this comment
a = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

# -----------------------------------------------------------------------------
# look at the variable explorer, what are the dimensions of both matrices
#  a:          b:      
# Can they be added together?
# -----------------------------------------------------------------------------
    
c1 = a + b

# look at c1 in your variable explorer, what happened?



# -----------------------------------------------------------------------------
# Change the dimensions of b to a 1 x 2 => lookup the reshape method
b = b.reshape(1,2)

c2 = a + b

# look at c2 in your variable explorer, what happened?



# -----------------------------------------------------------------------------
# Change the dimensions of b to a 2 x 1 => lookup the reshape method
b = b.reshape(2, 1)

c3 = a + b

# look at c3 in your variable explorer, what happened?



# -----------------------------------------------------------------------------
# Scalar addition
c4 = a + 2

# look at c4 in your variable explorer, what happended?



# -----------------------------------------------------------------------------
# when does Broadcasting not work?   specify a shape that results in a run time error



# =============================================================================
# Add a column of zeros to our matrix
#   - create a vector of zeros (#rows x 1)
#        note: determine the number of rows 2 ways
#              1. len of matrix
#              2. use the shape variable - will use later in this lab
#   - add (concatenate) to the beginning of the matrix
# =============================================================================
bias = np.zeros((c4.shape[0],1))
c4 = np.concatenate((bias, c4), axis=1)






# =============================================================================
# Slicing - Accessing a specific row or a column
# =============================================================================

# use the arange method to create an array of sequential data
A = np.arange(10,50)

# Look at the dimensions of A.   Reshape to a 4 x 10 matrix
A = A.reshape(4,10)

# -----------------------------------------------------------------------------
# store the first column into B, what is the shape of B?
B = A[:,0]

# -----------------------------------------------------------------------------
# store the 2nd row into C, what is the shape of C?
C = A[1,:]


# -----------------------------------------------------------------------------
# store the shape of C with the shape variable (not a method)
# note, shape is a tuple, an array that cannot be changed - static
dims =  C.shape 

# -----------------------------------------------------------------------------
# reshape to a 1 x #colms (using the tuple)
C = C.reshape(1, dims[0])


# -----------------------------------------------------------------------------
# use a loop to find the average value in C

# this uses matrices/lists
#sum = 0
#for i in range(len(C[0])):
#    sum += C[0][i]
#avg = sum/len(C[0])

# but we want to use numpy attributes
sum = 0
for i in range(dims[0]):
    sum += C[0,i]
avg = sum/dims[0]

# -----------------------------------------------------------------------------
# use numpy's average method and compare
nAvg = np.average(C)



# -----------------------------------------------------------------------------
# Use the average method with A, but specify an axis
# What happened with axis=0
# What happened with axis=1
nAAvg = np.average(A, axis=0)
nAAvg2 = np.average(A, axis=1)



# =============================================================================
# Slicing - Accessing multiple columns
# =============================================================================


#1. store the 1st, 3rd, 4th, 5th, and last column of A into a matrix D
D = A[:, [0,2,3,4,len(A)-1]]

# -----------------------------------------------------------------------------
#2. find the minimum values for each COLUMN using numpy's min method (axis=?)
colmin = np.mean(A, axis = 0)


# -----------------------------------------------------------------------------
#3. find the maximum values for each ROW using numpy's max method  (axis=?)
rowmax = np.max(A, axis = 1)

# -----------------------------------------------------------------------------
#4. create a new 5x8 matrix for D using random numbers with a range of 0 - 19.999999
Dn = np.random.random((5,8))*20
#Dn = np.arange(0,20, 0.5)
#Dn = Dn.reshape(5,8)

# =============================================================================
#5. find the standard deviation for each column in D 
#   - find the standard deviation in each column manually using the 
#     following numpy methods: average, square, sum, sqrt
# =============================================================================

 
std = np.sqrt((np.subtract(Dn, Dn.mean(axis = 0))**2).sum(axis = 0)/len(Dn))







# -----------------------------------------------------------------------------
#6. Use numpy's std method and compare in the variable explorer  (axis=?)
std2 = np.std(Dn, axis = 0)



