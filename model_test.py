# Matej Zecic
# CS5330 - Spring 2025
# Description: This script loads a pre-trained neural network and uses it to classify a set of custom handwritten digit
# images. The script loads the trained model, loads the custom images, preprocesses them to match the MNIST dataset, and
# runs the images through the network to obtain predictions. The script then prints the predictions and displays the images
# with the predictions.

import torch
import matplotlib.pyplot as plt
import numpy as np
from main import Net, test_loader, learning_rate, momentum  # existing imports from main.py
import torch.optim as optim

# Load your trained model
continued_network = Net()

# Load model state (weights)
network_state_dict = torch.load('results/model.pth')
continued_network.load_state_dict(network_state_dict)

# Create optimizer (not strictly needed for eval, but keeping for consistency)
continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate, momentum=momentum)

# (Optional) Load optimizer state (if you saved it)
# optimizer_state_dict = torch.load('optimizer.pth')
# continued_optimizer.load_state_dict(optimizer_state_dict)

# IMPORTANT: Set the network to evaluation mode
continued_network.eval()

# Get first batch of 10 images from test set
data_iter = iter(test_loader)
images, labels = next(data_iter)  # first 10 examples

# Run network on the images
outputs = continued_network(images)
predicted = torch.argmax(outputs, dim=1)

# Print detailed outputs clearly
print(f"{'Example':<8} {'Output Values (rounded)':<80} {'Predicted':<10} {'True Label'}")
for i in range(10):
    output_vals = outputs[i].detach().numpy()
    output_rounded = [f"{val:.2f}" for val in output_vals]
    print(f"{i:<8} {str(output_rounded):<80} {predicted[i].item():<10} {labels[i].item()}")

# Plot the first 9 predictions (3x3 grid)
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle('Predictions for First 9 Test Examples', fontsize=16)

for i, ax in enumerate(axes.flat):
    img = images[i].numpy().squeeze()
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Predicted: {predicted[i].item()}', fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('plot-predictions.png')
plt.show()
