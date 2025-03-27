# Matej Zecic
# CS5330 - Spring 2025
# Description: This script trains a neural network to classify Greek letters (alpha, beta, gamma) using a pre-trained
# MNIST network. The script loads the pre-trained MNIST network, freezes all weights, replaces the last layer with a new
# layer for the Greek letters, and trains the new layer on a custom Greek letters dataset. The script then tests the
# trained network on a set of handwritten Greek letters and prints the results.
#

import torch
import matplotlib.pyplot as plt
import torchvision
from main import Net, learning_rate, momentum
import os

# 1. Generate the MNIST network by importing from task 1 (done with the import above)
# 2. Read an existing model from a file and load pre-trained weights
network = Net()
network.load_state_dict(torch.load('results/model.pth'))

# 3. Freeze all of the network weights
for param in network.parameters():
    param.requires_grad = False

# 4. Replace the last layer with a new Linear layer with three nodes (for alpha, beta, gamma)
# Looking at neural_network.py, the last layer is fc2 which goes from 50 to 10 (digits)
# We need to replace it with a layer that goes from 50 to 3 (Greek letters)
network.fc2 = torch.nn.Linear(50, 3)

# Verify that we have trainable parameters (only in the last layer)
trainable_params = [p for p in network.parameters() if p.requires_grad]
print(f"Number of trainable parameters: {len(trainable_params)}")
assert len(trainable_params) > 0, "No trainable parameters! Check requires_grad settings."

# Optimizer for training only the new layer (fc2)
optimizer = torch.optim.SGD(network.fc2.parameters(), lr=learning_rate, momentum=momentum)

# Greek data set transform as specified in the requirements
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

# Path to the directory containing alpha, beta, gamma folders
training_set_path = 'data/greek_train'

# DataLoader for the Greek data set following the required format
greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(
        training_set_path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=5,
    shuffle=True
)

# Set up loss function
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# Testing function
def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# Train for multiple epochs to find how many it takes to reasonably identify the letters
num_epochs = 20
train_losses = []
train_accuracies = []

print("Starting training for Greek letters classification...")
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(network, greek_train, optimizer, criterion)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f"Epoch {epoch}/{num_epochs}: Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")

    # Early stopping if accuracy is very high
    if train_acc > 0.95:
        print(f"Reached high accuracy after {epoch} epochs. Stopping early.")
        break

# Save the trained model
torch.save(network.state_dict(), 'results/greek_model.pth')

# Plot the training progress
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('results/greek_training.png')
plt.show()

print(f"Training completed in {len(train_losses)} epochs.")
print(f"Final accuracy: {train_accuracies[-1]*100:.2f}%")
print("Model saved to 'results/greek_model.pth'")

# Function to classify a single image
def classify_greek_letter(image_path, model):
    from PIL import Image

    pil_image = Image.open(image_path)

    # Convert to RGB if the image has an alpha channel (RGBA)
    if pil_image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        # Paste the image using alpha as mask
        background.paste(pil_image, mask=pil_image.split()[3])
        pil_image = background
    elif pil_image.mode != 'RGB':
        # Convert any other mode to RGB
        pil_image = pil_image.convert('RGB')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = transform(pil_image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    classes = ['alpha', 'beta', 'gamma']
    return classes[predicted.item()]

# Testing on handwritten greek letters
greek_letters_dir = "data/handwritten_greek_letters"
test_images = []
ground_truth = []

for filename in os.listdir(greek_letters_dir):
    # Check if file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        test_images.append(os.path.join(greek_letters_dir, filename))

        # Extract letter for ground truth
        letter = filename.split('_')[0]
        ground_truth.append(letter)

# Create a single figure for all images
num_images = len(test_images)
cols = min(5, num_images)  # Maximum 5 images per row
rows = (num_images + cols - 1) // cols  # Calculate needed rows

plt.figure(figsize=(cols * 3, rows * 3))

# Track predictions and calculate accuracy
predictions = []
correct = 0

for i, (img_path, true_label) in enumerate(zip(test_images, ground_truth)):
    # Get prediction
    prediction = classify_greek_letter(img_path, network)
    predictions.append(prediction)

    # Check if prediction is correct
    is_correct = prediction == true_label
    if is_correct:
        correct += 1

    # Display the image with prediction
    plt.subplot(rows, cols, i + 1)
    img = plt.imread(img_path)
    plt.imshow(img)

    # Add color-coded title based on correctness
    color = 'green' if is_correct else 'red'
    plt.title(f"Pred: {prediction}\nTrue: {true_label}", color=color)
    plt.axis('off')

# Calculate accuracy
accuracy = correct / len(test_images) if test_images else 0
print(f"Overall accuracy: {accuracy:.2%} ({correct}/{len(test_images)})")

plt.tight_layout()
plt.savefig('results/greek_test_predictions.png')
plt.show()

# Print detailed results
print("\nDetailed results:")
for i, (img_path, true_label, pred) in enumerate(zip(test_images, ground_truth, predictions)):
    status = "✓" if true_label == pred else "✗"
    print(f"{i+1}. {os.path.basename(img_path)}: True: {true_label}, Pred: {pred} {status}")
