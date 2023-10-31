import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def predict(self, x):
        return 1 if np.dot(x, self.weights) + self.bias > 0 else 0

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                error = target - self.predict(x)
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

perceptron = Perceptron(input_size=2, learning_rate=0.1)

perceptron.train(X, y)

for i in range(len(X)):
    x = X[i]
    prediction = perceptron.predict(x)
    print(f"Input: {x}, Prediction: {prediction}")
