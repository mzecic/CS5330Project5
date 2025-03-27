# Matej Zecic
# CS5330 - Spring 2025
# Description: This script loads a pre-trained neural network and uses it to classify a set of custom handwritten digit
# images. The script loads the trained model, loads the custom images, preprocesses them to match the MNIST dataset, and
# runs the images through the network to obtain predictions. The script then prints the predictions and displays the images
# with the predictions.

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
from main import Net

# Path to your images folder
image_folder = 'data/images_handwritten'

# Load trained model clearly
model = Net()
model.load_state_dict(torch.load('results/model.pth'))
model.eval()

# MNIST normalization parameters
mnist_mean, mnist_std = 0.1307, 0.3081

# Define transforms to match MNIST
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),  # invert colors (MNIST digits are white-on-black)
    transforms.Normalize((mnist_mean,), (mnist_std,))
])

# Load and preprocess images clearly
image_files = sorted([file for file in os.listdir(image_folder) if file.endswith('.png')])
images_tensor_list = []

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    img = Image.open(img_path)
    img = transform(img)
    images_tensor_list.append(img)

# Stack images into tensor clearly
images_tensor = torch.stack(images_tensor_list)

# Run the network inference
outputs = model(images_tensor)
predicted = torch.argmax(outputs, dim=1)

# Print predictions clearly
print(f"{'Image':<20} {'Output Values (logits)':<80} {'Predicted Digit'}")
for i, img_file in enumerate(image_files):
    logits = outputs[i].detach().numpy()
    logits_rounded = [f"{val:.2f}" for val in logits]
    print(f"{img_file:<20} {str(logits_rounded):<80} {predicted[i].item()}")

# Plot your images with predictions clearly
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Predictions on Your Handwritten Digit Images', fontsize=16)

for i, ax in enumerate(axes.flat):
    # Convert tensor to image for display (undo normalization clearly)
    img = images_tensor[i].numpy().squeeze() * mnist_std + mnist_mean
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Predicted: {predicted[i].item()}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('custom_images_predictions.png')
plt.show()
