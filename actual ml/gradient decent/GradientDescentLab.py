"""
Name: Akhil Gogineni
    
Please:
    - set breakpoints and walk through this lab.  
    - verify the dimensions of the arrays
    - verify values with your hand calculations
    - seek to understand, not to just get it done
    
Lab: Gradient Descent
  - use genFirstDataSet() to get your Cost and GradientDescent functions to work
  - then use genDataSet() to get a random set of data with a given variance (sigma)
  
ToDo:
    - plot random data (20 values)
    - plot your predicted line given w0 and w1
    - plot the cost curve
    - answer the questions in the lab in the commented area at the end
    - congratulate yourself - you passed a major milestone!
    
"""
import numpy as np
import matplotlib.pyplot as plt


def genFirstDataSet():
    return np.array([[1, 0, 0],
                     [1, 1, .5],
                     [1, 2, 1],
                     [1, 3, 1.5]])


def genDataSet(nElements, sigma, slope, offset, minX, maxX):
    x0 = np.ones(nElements)
    x1 = np.random.random(nElements) * (maxX - minX) + minX
    y = x1 * slope + offset
    y += (np.random.random(nElements) * 2 - 1) * sigma

    data = np.column_stack((x0, x1, y))
    return data


def costFunc(x, weights, y):
    # todo: compute the cost using numpy, not for-loops
    predvals = weights[1, 0]*x[:, 1]+weights[0, 0]
    
    sqerr = np.square(predvals - y)
    merror = np.mean(sqerr, axis = 0)
    return merror
    

        
def gradDesc(x, weights, y):
    predvals = weights[1,0]*x[:,1]+weights[0,0]
    err = predvals-y
    
    grad = np.dot(err.T,x)
    return grad




#data = genFirstDataSet()
data = genDataSet(20, 2, 2, 0, 0, 20)
arraylen = len(data)

#todo:  using the split function with data to separate x and y
x= data[:,0:2]
y= data[:, 2]


weights = np.array([0, 1.5]).reshape(2, 1)


# LR - learning rate,  change to .005 when using generated data set.
LR = 0.0005
costArray = []
gradArray = []
weightArray = []

maxIter = 50

for i in range(maxIter):
    
    
    cost = costFunc(x, weights, y)
    costArray.append(cost)
        
    gradients = gradDesc(x, weights, y)
    gradArray.append(gradients)
    
    #update weights 
    
    weights = weights - (LR*(gradients.reshape(2,1)))
    weightArray.append(weights)
    
    #print((gradients))
    #print(weights[1])
    
        
iters = np.arange(0,maxIter)


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

ax1.plot(xVals, predictedY, linewidth=1, linestyle='-', color='Red')


# plot cost
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(maxIter), costArray, color='blue')
ax2.set(title = 'Cost vs. Iterations', xlabel = 'iterations', ylabel = 'cost')


'''
Compare and Contrast

ToDo: answer questions in the lab here.
  
1) Increase the learning rate. At what point does the model fail to train?
    The model fails at 0.005.

2) Decrease the learning rate.  What happens to the Cost Function graph?
    It becomes less flat and more into a straight line with a negative slope.

3) Display the final weights and compare with the expected values.  What can 
    you change to make your model more accurate?
    
    Lower the learning rate.

'''