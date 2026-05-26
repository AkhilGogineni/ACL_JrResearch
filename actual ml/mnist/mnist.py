#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:49:53 2022

@author: akhilgogineni
"""

from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy import stats as st



(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

image_index = 10002 # You may select anything up to 60,000
print(train_labels[image_index]) 
one_image = train_images[image_index]


print ('Train Shape', train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)


test_images = test_images.reshape((10000, 28*28))
train_images = train_images.reshape((60000, 28*28))

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
one_image = one_image.astype('float32')/255

before_categ_test_labels = test_labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()

network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))

network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer = 'rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print(network.summary())


# =============================================================================
# Create predictions and probabilities using test dataset
# =============================================================================
prediction = network.predict(test_images)
# =============================================================================
# Create an array of the max prob for each image
# =============================================================================
digit_prediction = np.argmax (prediction, axis = 1) 
digit_test_labels = np.argmax(test_labels, axis = 1) 
difference = digit_test_labels - digit_prediction 
number_wrong = difference[difference[:]!=0]
# =============================================================================
# Stack the predictions, actuals and probabilities with images
# =============================================================================
images_with_probab = np.column_stack((prediction, digit_prediction, digit_test_labels, test_images))
# =============================================================================
# Choose wrong and correct predictions
# Sort by probabilities
# Choose the 6 highest probabilities of both
# =============================================================================
print((images_with_probab[:, 10:11] == images_with_probab[:, 11:12]).shape, images_with_probab. shape)
correct_array = images_with_probab[:,:][(images_with_probab[:, 10:11] == images_with_probab[:, 11:12])[:,0]]
wrong_array = images_with_probab[:, :][(images_with_probab[:, 10:11] != images_with_probab[:, 11:12])[:,0]]


wrong_array_copy = wrong_array 
worst_images_info =[] 
for i in range (6):
    wrong_index = np.argmax(wrong_array_copy[:, 0:10], axis = 0)
    worst_images_info.append(wrong_array_copy[wrong_index, :])
    wrong_array_copy = np.delete(wrong_array_copy, wrong_index, axis = 0)

# =============================================================================
# Plot the six worst and six most accurate predictions
# =============================================================================


x = network.predict(test_images)
def showWorst():
    wrong_indices = []
    wrong_vals = []
    for i in range(len(x)):
        temp = np.argmax(x[i],axis=0)
        temp2 = np.argmax(test_labels, axis=1) 
        if (temp!=temp2[i]):
            wrong_indices.append(i)
            wrong_vals.append(x[i][temp])
            
    indices_copy = np.copy(wrong_indices)
    worst =[]
    wrong_indices = np.argsort (wrong_vals, axis=0)
    wrong_indices = wrong_indices[::-1]
    for i in range (6):
        worst.append(indices_copy[wrong_indices[i]])
        print('Actual: ', np.argmax(test_labels[worst[i]]))
        print('Predicted: ', np.argmax(x[worst[i]],axis=0))
        print()

    fig, ax = plt. subplots (2, 3)
    fig.suptitle('Worst incorrect predictions')
        
    for i, ax in enumerate(ax.flatten()):
        p = np.argmax(x[worst[i]]).astype(int).astype(str)
        a = np.argmax(test_labels[worst[i]]).astype(int).astype(str)
        ax.set_title('P: '+p+' A: '+a) 
        image = test_images [worst [i]]
        plottable_image = np.reshape(image, (28, 28))
        ax.imshow(plottable_image, cmap= 'gray_r')
             
        
def showBest():
    x = network.predict(test_images)
    wrong_indices = []
    wrong_vals = []
    for i in range(len(x)):
        temp = np.argmax(x[i],axis=0)
        temp2 = np.argmax(test_labels,axis=1)
        if (temp==temp2[i]):
            wrong_indices.append(i)
            wrong_vals.append(x[i][temp])
    
    indices_copy = np. copy (wrong_indices)
    worst = [1]
    wrong_indices = np.argsort(wrong_vals, axis = 0)
    wrong_indices = wrong_indices[::-1]

    for i in range(6):
        worst.append(indices_copy[wrong_indices[i]])
    
    fig, ax = plt. subplots(2, 3)
    fig.suptitle('Best Correct predictions') 
    for i, ax in enumerate (ax. flatten()) :
            p = np.argmax(x[worst [i]]) . astype(int).astype(str)
            a = np.argmax(test_labels [worst [1]]).astype(int).astype(str)
            ax.set_title('P: ' + p + ' A: ' +a)
            image = test_images [worst[1]]
            plottable_image = np.reshape (image, (28, 28)) 
            ax.imshow (plottable_image, cmap= 'gray_r')



        
showWorst()
showBest()
                  
        
count = 0
for i in range(digit_test_labels.shape[0]) :
    column = digit_test_labels[i]
    if correct_array[1, column] > 0.999:
        count+= 1
        
                  
        
                  
        
                  
        
                  
        
                  
        
                  
        
                  
        
                  
        
                  
        
                  