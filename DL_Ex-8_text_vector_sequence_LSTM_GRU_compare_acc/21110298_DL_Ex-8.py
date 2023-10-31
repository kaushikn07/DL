import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
import time

num_words = 10000  
(x_train_text, y_train), (x_test_text, y_test) = imdb.load_data(num_words=num_words)


max_sequence_length = 200  
x_train_text = pad_sequences(x_train_text, maxlen=max_sequence_length)
x_test_text = pad_sequences(x_test_text, maxlen=max_sequence_length)

sequence_length = 200
num_features = 50
x_train_vector = np.random.rand(len(x_train_text), sequence_length, num_features)
x_test_vector = np.random.rand(len(x_test_text), sequence_length, num_features)


def create_lstm_model():
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=128, input_length=max_sequence_length),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_gru_model():
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=128, input_length=max_sequence_length),
        GRU(64),
        Dense(1, activation='sigmoid')
    ])
    return model

lstm_model = create_lstm_model()
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
lstm_model.fit(x_train_text, y_train, epochs=3, batch_size=128, validation_data=(x_test_text, y_test))
lstm_time = time.time() - start_time

test_loss, test_accuracy = lstm_model.evaluate(x_test_text, y_test)
print(f"LSTM Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"LSTM Training Time: {lstm_time:.2f} seconds")

gru_model = create_gru_model()
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
gru_model.fit(x_train_text, y_train, epochs=3, batch_size=128, validation_data=(x_test_text, y_test))
gru_time = time.time() - start_time

test_loss, test_accuracy = gru_model.evaluate(x_test_text, y_test)
print(f"GRU Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"GRU Training Time: {gru_time:.2f} seconds")
