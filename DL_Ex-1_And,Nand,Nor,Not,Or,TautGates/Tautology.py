import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [1], [1], [1]])

model = Sequential()
model.add(Dense(2, activation='relu', input_shape=(2,)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=10000, batch_size=1, verbose=0)

predictions = model.predict(X)
print("Input: ", X)
print("Predictions: ", predictions)
