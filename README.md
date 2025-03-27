# Neural Network Image Recognition Project

## Author
Matej Zecic

## Project Overview
This project implements neural networks for image recognition tasks with PyTorch, focusing on MNIST digit classification and extending to Greek letters and Fashion MNIST. It explores different neural network architectures and hyperparameters to optimize performance.

## Contents

### Part 1: MNIST Digit Classification
- Implementation of a CNN for handwritten digit recognition
- Training and evaluation on the MNIST dataset
- Analysis of model performance and visualization of training/testing losses

### Part 2: Network Analysis and Visualization
- Visualization of convolutional filters in the trained network
- Application of these filters to input images
- Visual explanation of feature detection

### Part 3: Transfer Learning for Greek Letters
- Adapting the pre-trained MNIST network for Greek letter classification (alpha, beta, gamma)
- Freezing weights and replacing the output layer
- Training and evaluation on custom Greek letter dataset

### Part 4: Network Architecture Experimentation
- Systematic exploration of neural network architectures for Fashion MNIST
- Optimization across 6 dimensions:
  1. Number of convolutional layers
  2. Filter sizes and counts
  3. Kernel sizes
  4. Fully connected layer configurations
  5. Dropout rates
  6. Activation functions
- Performance analysis and visualization of results

### Part 5: Testing on Custom Images
- Application of trained models to custom handwritten digits
- Preprocessing and normalization for compatibility
- Visualization and analysis of predictions

## Key Files
- main.py: Main MNIST training implementation
- neural_network.py: Neural network architecture definition
- analyze_network.py: Visualization of filters and feature maps
- greek_letters.py: Transfer learning for Greek letter recognition
- network_experiment_fashion.py: Architecture experimentation on Fashion MNIST
- model_test_handwritten.py: Testing on custom handwritten digits

## Results
Training resulted in high accuracy models for both MNIST digits and Greek letters. The Fashion MNIST experiments identified optimal hyperparameters with a final test accuracy of 90.15% using a single convolutional layer with large filters (32, 64), LeakyReLU activation, and a higher learning rate (0.1).

## Extensions
- Extended the architecture experimentation from 3 to 6 dimensions, exploring:
  - Number of convolutional layers
  - Filter configurations
  - Kernel sizes
  - Hidden layer configurations
  - Dropout strategies
  - Activation functions

## Data
All training data is downloaded automatically by the scripts except for the custom handwritten digits and Greek letters which are stored in the data directory.

## Links/URLs
- No external links or videos submitted with this project.

## Usage
1. Run main.py to train the base MNIST model
2. Run analyze_network.py to visualize the network filters
3. Run greek_letters.py to perform transfer learning
4. Run network_experiment_fashion.py to experiment with architectures
5. Run model_test_handwritten.py to test on custom images

## Time Travel Days
Not using any time travel days.
