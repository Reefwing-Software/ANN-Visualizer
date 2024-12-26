# Copyright (c) 2024 David Such
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model

# Load the neural network model
model_path = Path("~/Documents/GitHub/DNN-Visualizer/visualdnn/src/visualdnn/model/mnist_model.h5").expanduser()

if model_path.exists():  
    model = load_model(model_path)
    print("Loaded the pre-trained model.")
else:
    print("Model file not found. Please train the model first.")
    sys.exit()

# Access the weights of the first and second hidden layers
weights_layer_1, biases_layer_1 = model.layers[0].get_weights()  # First hidden layer
weights_layer_2, biases_layer_2 = model.layers[1].get_weights()  # Second hidden layer

# Create subplots to compare the weight strengths of both layers
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Plot the weight connections of the first hidden layer
im1 = axes[0].imshow(weights_layer_1, aspect='auto', cmap='viridis')
axes[0].set_title("Neuron Connections - Input to Layer 1")
axes[0].set_xlabel("Neurons in Layer 1")
axes[0].set_ylabel("Input Features (Flattened 28x28)")
plt.colorbar(im1, ax=axes[0])

# Plot the weight connections of the second hidden layer
im2 = axes[1].imshow(weights_layer_2, aspect='auto', cmap='viridis')
axes[1].set_title("Neuron Connections - Layer 1 to Layer 2")
axes[1].set_xlabel("Neurons in Layer 2")
axes[1].set_ylabel("Neurons in Layer 1")
plt.colorbar(im2, ax=axes[1])

# Adjust layout and show the plots
plt.tight_layout()
plt.show()