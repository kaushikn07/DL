import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_dim, hidden_dim, output_dim):
    np.random.seed(0)
    weights_input_hidden = np.random.uniform(size=(input_dim, hidden_dim))
    weights_hidden_output = np.random.uniform(size=(hidden_dim, output_dim))
    return weights_input_hidden, weights_hidden_output

def forward_propagation(X, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

def backpropagation(X, y, output, hidden, weights_hidden_output):
    error = y - output
    output_delta = error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden)

    return output_delta, hidden_delta

def update_weights(X, hidden, output_delta, hidden_delta, learning_rate, weights_input_hidden, weights_hidden_output):
    weights_hidden_output += hidden.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

def train(X, y, epochs=10000, learning_rate=0.1):
    input_dim = X.shape[1]
    hidden_dim = 2
    output_dim = 1

    weights_input_hidden, weights_hidden_output = initialize_weights(input_dim, hidden_dim, output_dim)

    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation(X, weights_input_hidden, weights_hidden_output)

        output_delta, hidden_delta = backpropagation(X, y, output_layer_output, hidden_layer_output, weights_hidden_output)

        update_weights(X, hidden_layer_output, output_delta, hidden_delta, learning_rate, weights_input_hidden, weights_hidden_output)

    return weights_input_hidden, weights_hidden_output

def predict(X, weights_input_hidden, weights_hidden_output):
    _, output_layer_output = forward_propagation(X, weights_input_hidden, weights_hidden_output)
    return np.round(output_layer_output)

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1], [0], [0], [1]])

    weights_input_hidden, weights_hidden_output = train(X, y)

    for i in range(X.shape[0]):
        x_input = X[i].reshape(1, -1)
        prediction = predict(x_input, weights_input_hidden, weights_hidden_output)
        print(f"Input: {x_input}, Predicted Output: {prediction}")
