# Copyright (c) 2024 David Such
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def show_images(images, titles):
    """Display images with their corresponding titles."""
    num_images = len(images)
    cols = 5
    rows = (num_images + cols - 1) // cols  # Calculate number of rows needed

    plt.figure(figsize=(15, rows * 3))
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')  # Show in grayscale
        plt.title(title, fontsize=8)

        # Add gridlines
        ax.set_xticks(range(0, 28, 5))  # Set x-axis grid spacing (for 28x28 images)
        ax.set_yticks(range(0, 28, 5))  # Set y-axis grid spacing
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)  # Light gray dashed gridlines
    plt.tight_layout()
    plt.show()

# Show some random training and test images
images_to_show = []
titles_to_show = []

# Add random training images
for _ in range(10):
    r = random.randint(0, len(x_train) - 1)
    images_to_show.append(x_train[r])
    titles_to_show.append(f"Training Image [{r}] = {y_train[r]}")

# Add random test images
for _ in range(5):
    r = random.randint(0, len(x_test) - 1)
    images_to_show.append(x_test[r])
    titles_to_show.append(f"Test Image [{r}] = {y_test[r]}")

# Display the images
show_images(images_to_show, titles_to_show)