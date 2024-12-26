# Copyright (c) 2024 David Such
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input

# Load the neural network model
model_path = Path("~/Documents/GitHub/DNN-Visualizer/visualdnn/src/visualdnn/model/mnist_model.keras").expanduser()

if model_path.exists():  
    model = load_model(model_path)
    print("Loaded the pre-trained model.")
else:
    print("Model file not found. Please train the model first.")
    sys.exit()

layer_1_model = Sequential([
    Input(shape=(784,)),
    Dense(25, activation='relu', name='hidden_layer_1')
], name="hidden_layer_1_model") 

# Set weights of the first layer
hidden_layer_1_weights = model.get_layer('hidden_layer_1').get_weights()
layer_1_model.get_layer('hidden_layer_1').set_weights(hidden_layer_1_weights)
layer_1_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
layer_1_model.summary()

layer_2_model = Sequential([
    Input(shape=(784,)),
    Dense(25, activation='relu', name='hidden_layer_1'),  
    Dense(25, activation='relu', name='hidden_layer_2')  
], name="hidden_layer_2_model") 

hidden_layer_2_weights = model.get_layer('hidden_layer_2').get_weights()
layer_2_model.get_layer('hidden_layer_1').set_weights(hidden_layer_1_weights)
layer_2_model.get_layer('hidden_layer_2').set_weights(hidden_layer_2_weights)
layer_2_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
layer_2_model.summary()

layer_1_model.save("hidden_layer_1_model.keras")
layer_2_model.save("hidden_layer_2_model.keras")

print("Hidden level models saved successfully.")

# Determine the output range for the hidden layers
min_input = np.zeros((1, 784))  # Minimum input: all zeros
max_input = np.ones((1, 784))  # Maximum input: all ones

# Get predictions for the minimum input
layer_1_min_output = layer_1_model.predict(min_input)
layer_2_min_output = layer_2_model.predict(min_input)

# Get predictions for the maximum input
layer_1_max_output = layer_1_model.predict(max_input)
layer_2_max_output = layer_2_model.predict(max_input)

# Print the results
print("First Hidden Layer Outputs:")
print(f"Minimum input prediction: Min={layer_1_min_output.min()}, Max={layer_1_min_output.max()}")
print(f"Maximum input prediction: Min={layer_1_max_output.min()}, Max={layer_1_max_output.max()}")

print("\nSecond Hidden Layer Outputs:")
print(f"Minimum input prediction: Min={layer_2_min_output.min()}, Max={layer_2_min_output.max()}")
print(f"Maximum input prediction: Min={layer_2_max_output.min()}, Max={layer_2_max_output.max()}")

# Calculate the mean and range (min and max) of activations for each neuron
layer_1_means = np.mean([layer_1_min_output, layer_1_max_output], axis=0).flatten()
layer_1_ranges = np.abs(layer_1_max_output - layer_1_min_output).flatten()

layer_2_means = np.mean([layer_2_min_output, layer_2_max_output], axis=0).flatten()
layer_2_ranges = np.abs(layer_2_max_output - layer_2_min_output).flatten()

# Plot the activations for the two hidden layers
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Hidden Layer 1
axes[0].errorbar(
    range(1, 26),  # Neuron indices
    layer_1_means,  # Mean activations
    yerr=layer_1_ranges / 2,  # Error bars (half the range)
    fmt='o',
    ecolor='lightblue',
    capsize=4,
    label='Neuron Activation Range'
)
axes[0].set_title('Hidden Layer 1 Activation Range')
axes[0].set_xlabel('Neuron Index')
axes[0].set_ylabel('Activation Value')
axes[0].grid(True)
axes[0].legend()

# Plot for Hidden Layer 2
axes[1].errorbar(
    range(1, 26),  # Neuron indices
    layer_2_means,  # Mean activations
    yerr=layer_2_ranges / 2,  # Error bars (half the range)
    fmt='o',
    ecolor='lightcoral',
    capsize=4,
    label='Neuron Activation Range'
)
axes[1].set_title('Hidden Layer 2 Activation Range')
axes[1].set_xlabel('Neuron Index')
axes[1].set_ylabel('Activation Value')
axes[1].grid(True)
axes[1].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()