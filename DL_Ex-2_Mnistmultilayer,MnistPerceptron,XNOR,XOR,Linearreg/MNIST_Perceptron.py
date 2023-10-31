import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
plt.imshow(X_train[0], cmap='gray')
plt.title("Label: " + str(y_train[0]))
plt.show()
X_train = X_train / 255.0
X_test = X_test / 255.0
model = Sequential()
model.add(Dense(10, activation='softmax', input_shape=(28 * 28,)))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
X_train_flat = X_train.reshape(-1, 28 * 28)
X_test_flat = X_test.reshape(-1, 28 * 28)
model.fit(X_train_flat, y_train, epochs=10, batch_size=32, validation_split=0.1)
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test)
print("Test Accuracy:", test_accuracy)





