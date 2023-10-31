import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models


data = pd.read_csv('wine.csv')


x = data.drop('quality', axis=1).values
y = data['quality'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def create_nn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(num_classes, activation='tanh')) 
    return model

input_shape = x_train.shape[1]
num_classes = len(np.unique(y))


model = create_nn_model(input_shape, num_classes)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()


model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy*100:.2f}%')

#In this tweeked architecture code I have increased the number of layers: The neural network now has four hidden layers instead of two, with 128, 64, 32, and 16 neurons, respectively, and I have Changed the activation function. I am using the using the ReLU activation function for all hidden layers.

#Original Model ~63.25%
#Modified Model (Increased Layers) ~77-65%
#Modified Model (ReLU Activation)~77-65%

