import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from sklearn.metrics import confusion_matrix, classification_report

# Load and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define a function to create and train a CNN model
def train_cnn_model(num_filters, filter_size, pool_size, num_fc_units):
    model = models.Sequential()
    model.add(layers.Conv2D(num_filters, filter_size, activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_fc_units, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels), verbose=2)

    return model, history

# Define a function to evaluate a model and generate a confusion matrix
def evaluate_model(model):
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    # Get predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_labels)

    print("Test accuracy:", test_acc)
    print("Confusion Matrix:")
    print(conf_matrix)

# Train and evaluate baseline model
baseline_model, _ = train_cnn_model(num_filters=32, filter_size=(3, 3), pool_size=(2, 2), num_fc_units=128)
print("Baseline Model:")
evaluate_model(baseline_model)

# Ablation Study 1: Vary the number of filters
num_filters_list = [16, 32, 64]
for num_filters in num_filters_list:
    model, _ = train_cnn_model(num_filters=num_filters, filter_size=(3, 3), pool_size=(2, 2), num_fc_units=128)
    print(f"Ablation Study: Number of Filters = {num_filters}")
    evaluate_model(model)

# Ablation Study 2: Vary the filter size
filter_size_list = [(3, 3), (5, 5)]
for filter_size in filter_size_list:
    model, _ = train_cnn_model(num_filters=32, filter_size=filter_size, pool_size=(2, 2), num_fc_units=128)
    print(f"Ablation Study: Filter Size = {filter_size}")
    evaluate_model(model)

# Ablation Study 3: Vary the pooling size
pool_size_list = [(2, 2), (3, 3)]
for pool_size in pool_size_list:
    model, _ = train_cnn_model(num_filters=32, filter_size=(3, 3), pool_size=pool_size, num_fc_units=128)
    print(f"Ablation Study: Pooling Size = {pool_size}")
    evaluate_model(model)

# Ablation Study 4: Vary the number of fully connected units
num_fc_units_list = [64, 128, 256]
for num_fc_units in num_fc_units_list:
    model, _ = train_cnn_model(num_filters=32, filter_size=(3, 3), pool_size=(2, 2), num_fc_units=num_fc_units)
    print(f"Ablation Study: Number of Fully Connected Units = {num_fc_units}")
    evaluate_model(model)


