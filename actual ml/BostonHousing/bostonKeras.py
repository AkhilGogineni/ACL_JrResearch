#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:55:18 2023

@author: akhilgogineni
"""

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler

def calcMAE(pred, act):
    return np.mean(abs(pred-act))
  
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# =============================================================================
# Standardize your input
# =============================================================================
sc = StandardScaler()
# =============================================================================
#     # Standardize the training dataset
#     #(and calculate the mean and standard deviation)
# =============================================================================
train_data = sc.fit_transform(train_data)
# =============================================================================
#     Use this mean and standard deviation
#     calculated in the training dataset to
#     standardize the test dataset
# =============================================================================
test_data = sc.transform(test_data)

network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(13,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(1))
network.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = network.fit(train_data, train_targets, epochs=100, batch_size=1)

mae_history = history.history['mae']
test_mse_score, test_mae_score = network.evaluate(test_data, test_targets)

predicted_prices = network.predict(test_data)

print("My calculated MAE was: " + str(calcMAE(predicted_prices, test_targets.reshape(test_targets.shape[0], 1))))
print("The Model's MAE was: " + str(test_mae_score))

fig1 = plt.figure()
plt.title("Epochs v.s. MAE Keras Dataset", loc='center')
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.plot(range(1, 101), mae_history)
