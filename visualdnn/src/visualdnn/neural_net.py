# Copyright (c) 2024 David Such
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model

def visualize_neural_network(weights_layer_1, weights_layer_2):
    """
    Visualize two hidden layers as nodes with weighted connections.
    
    Args:
        weights_layer_1: Weight matrix of the first hidden layer (shape: [input_nodes, hidden_nodes_1]).
        weights_layer_2: Weight matrix of the second hidden layer (shape: [hidden_nodes_1, hidden_nodes_2]).
    """
    # Define node positions
    input_nodes = weights_layer_1.shape[0]
    hidden_nodes_1 = weights_layer_1.shape[1]
    hidden_nodes_2 = weights_layer_2.shape[1]

    # X-axis positions for each layer
    x_positions = [0, 1, 2]  # Input layer, Hidden Layer 1, Hidden Layer 2

    # Y-axis positions for nodes in each layer
    y_input = np.linspace(-1, 1, input_nodes)
    y_hidden_1 = np.linspace(-1, 1, hidden_nodes_1)
    y_hidden_2 = np.linspace(-1, 1, hidden_nodes_2)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot nodes
    ax.scatter([x_positions[0]] * input_nodes, y_input, s=50, label="Input Layer")
    ax.scatter([x_positions[1]] * hidden_nodes_1, y_hidden_1, s=100, label="Hidden Layer 1")
    ax.scatter([x_positions[2]] * hidden_nodes_2, y_hidden_2, s=100, label="Hidden Layer 2")

    # Plot connections between Input -> Hidden Layer 1
    for i in range(input_nodes):
        for j in range(hidden_nodes_1):
            weight = weights_layer_1[i, j]
            ax.plot(
                [x_positions[0], x_positions[1]],
                [y_input[i], y_hidden_1[j]],
                color=plt.cm.viridis(abs(weight)),  # Color based on weight strength
                linewidth=2 * abs(weight),  # Line thickness based on weight strength
                alpha=0.7
            )

    # Plot connections between Hidden Layer 1 -> Hidden Layer 2
    for i in range(hidden_nodes_1):
        for j in range(hidden_nodes_2):
            weight = weights_layer_2[i, j]
            ax.plot(
                [x_positions[1], x_positions[2]],
                [y_hidden_1[i], y_hidden_2[j]],
                color=plt.cm.viridis(abs(weight)),  # Color based on weight strength
                linewidth=2 * abs(weight),  # Line thickness based on weight strength
                alpha=0.7
            )

    # Set plot limits and labels
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Input Layer", "Hidden Layer 1", "Hidden Layer 2"])
    ax.set_yticks([])
    ax.legend()
    ax.set_title("Visualization of Neural Network Weights")
    plt.show()


# Load the neural network model
model_path = Path("~/Documents/GitHub/DNN-Visualizer/visualdnn/src/visualdnn/model/mnist_model.h5").expanduser()

if model_path.exists():  
    model = load_model(model_path)
    print("Loaded the pre-trained model.")
else:
    print("Model file not found. Please train the model first.")
    sys.exit()

# Example Usage: Assuming model.layers[0] and model.layers[1] are the hidden layers
weights_layer_1, _ = model.layers[0].get_weights()  # Input -> Hidden Layer 1
weights_layer_2, _ = model.layers[1].get_weights()  # Hidden Layer 1 -> Hidden Layer 2

visualize_neural_network(weights_layer_1, weights_layer_2)