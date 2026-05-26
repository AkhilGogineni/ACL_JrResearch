#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:39:40 2023

@author: akhilgogineni
"""
from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

image_index = 703  # You may select anything up to 60,000
print(train_labels[image_index])  # The label is 8
plt.imshow(train_images[image_index], cmap='Greys')
one_image = train_images[image_index]

print('Train Shape', train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)

test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.reshape((60000, 28, 28, 1))

train_images = train_images.astype('float')/255
test_images = test_images.astype('float')/255
one_image = one_image.astype('float')/255

before_categ_test_labels = test_labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Conv2D(
    32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))

network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print(network.summary())

