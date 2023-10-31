import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

X_train = np.linspace(0, 1, 100)
y_train = 2 * X_train + 1 + np.random.uniform(-0.5, 0.5, 100)

learning_rate = 0.01
training_epochs = 1000

X_train_tf = tf.constant(X_train, dtype=tf.float32)
y_train_tf = tf.constant(y_train, dtype=tf.float32)

W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

def linear_regression(x):
    return W * x + b

def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

optimizer = tf.keras.optimizers.SGD(learning_rate)

for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        predictions = linear_regression(X_train_tf)
        loss = mean_squared_error(y_train_tf, predictions)
    
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{training_epochs}, Loss: {loss:.4f}")

final_loss = mean_squared_error(y_train_tf, linear_regression(X_train_tf)).numpy()
final_weight = W.numpy()
final_bias = b.numpy()

print("Final Training Cost:", final_loss)
print("Final Weight:", final_weight)
print("Final Bias:", final_bias)
