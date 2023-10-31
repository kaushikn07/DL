import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


data = pd.read_csv("wine.csv")
x, y = data.data, data.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Weight Decay
def create_nn_model(input_shape, num_classes, l2_lambda=0.01):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='sigmoid', input_shape=(input_shape,), kernel_regularizer=l2(l2_lambda)))
    model.add(layers.Dense(32, activation='sigmoid', kernel_regularizer=l2(l2_lambda)))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

l2_lambda = 0.01 
model = create_nn_model(input_shape, num_classes, l2_lambda)


input_shape = x_train.shape[1]
num_classes = len(np.unique(y))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()


model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.1)


loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy*100:.2f}%')

