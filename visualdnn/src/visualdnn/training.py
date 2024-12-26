# Copyright (c) 2024 David Such
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the input data into 1D arrays
x_train = x_train.reshape(-1, 784)  # Training data shape: (60000, 784)
x_test = x_test.reshape(-1, 784)    # Test data shape: (10000, 784)

# Convert the target labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define the ANN model
model = Sequential([
    Input(shape=(784,)),
    Dense(25, activation='relu', name='hidden_layer_1'),  
    Dense(25, activation='relu', name='hidden_layer_2'),  
    Dense(10, activation='softmax', name='output_layer')
], name="mnist_classifier_model") 

# Print model summary
model.summary()

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Save the model to a file
model.save('mnist_model.keras')

# Create subplots for accuracy and loss plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot training and validation accuracy
axes[0].plot(history.history["accuracy"], label="Training Accuracy", marker='o')
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", marker='o')
axes[0].set_title("Accuracy Over Epochs")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.5)

# Plot training and validation loss
axes[1].plot(history.history["loss"], label="Training Loss", marker='o')
axes[1].plot(history.history["val_loss"], label="Validation Loss", marker='o')
axes[1].set_title("Loss Over Epochs")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.5)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()