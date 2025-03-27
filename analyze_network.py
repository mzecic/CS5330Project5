# Matej Zecic
# CS5330 - Spring 2025
# Description: This script loads a pre-trained neural network and displays the first layer's filters and their application
# to the first image in the training set. The script loads the trained model, extracts the first layer's weights, and
# applies them to the first image in the training set. The script then displays the filters and the results of applying
# the filters to the image.

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from neural_network import Net
import cv2
from main import train_loader

# Load trained model
model = Net()
network_state_dict = torch.load('results/model.pth')
model.load_state_dict(network_state_dict)

print(model)
weights = model.conv1.weight[:10]

for i in range(len(weights)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(weights[i][0].detach().numpy(), cmap='gray')
    plt.title(f'Filter {i}')
    plt.axis('off')

plt.show()

# Load first image from a training set and apply first layer's 10 filters

with torch.no_grad():
    first_training_image = next(iter(train_loader))[0][0].unsqueeze(0)

    # Create a figure with subplots
    plt.figure(figsize=(15, 10))

    for i, weight in enumerate(weights):
        # Extract the 2D weight kernel (removing the channel dimension)
        weight_2d = weight[0].detach().numpy()

        # Apply filter using the 2D kernel
        output = cv2.filter2D(first_training_image.squeeze().numpy(), -1, weight_2d)

        # Display filter
        plt.subplot(5, 4, 2*i + 1)
        plt.imshow(weight_2d, cmap='gray')
        plt.title(f'Filter {i}')
        plt.axis('off')

        # Display result
        plt.subplot(5, 4, 2*i + 2)
        plt.imshow(output, cmap='gray')
        plt.title(f'Result {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
