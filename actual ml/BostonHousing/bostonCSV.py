"""
Created on Mon Jan 23 22:00:18 2023

@author: akhilgogineni
"""
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.preprocessing import StandardScaler


def readDataFile(name):
    fileName = name
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    # loadtxt defaults to floats
    data = np.loadtxt(raw_data, dtype='str', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                      skiprows=1, delimiter=",")
    x = data[:, range(12)]
    y = data[:, 12]
    y = y.astype(np.float64)
    return x, y


def calcMAE(pred, act):
    return np.mean(abs(pred-act))


x, y = readDataFile("boston.csv")
sc = StandardScaler()

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, train_size=0.75, test_size=0.25, random_state=101)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# =============================================================================
# Standardize your input
# =============================================================================

network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(12,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(1))
network.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = network.fit(x_train, y_train,
                    epochs=100, batch_size=1)

mae_history = history.history['mae']
test_mse_score, test_mae_score = network.evaluate(x_test, y_test)
predicted_prices = network.predict(x_test)

print("My calculated MAE was: " +
      str(calcMAE(predicted_prices, y_test.reshape(y_test.shape[0], 1))))
print("The Model's MAE was: " + str(test_mae_score))

fig1 = plt.figure()
plt.title("Epochs v.s. MAE CSV Dataset", loc='center')
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.plot(range(1, 101), mae_history)
