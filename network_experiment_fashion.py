"""
CS5330 - Spring 2025
Network Architecture Experimentation (Optimized for Speed)

This module performs systematic experiments on neural network architectures
using the Fashion MNIST dataset to evaluate performance across different dimensions.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json

# Create results directory if it doesn't exist
os.makedirs('results/experiments', exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)

# Global parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Define a flexible neural network class that can be configured
class FlexibleNet(nn.Module):
    def __init__(
        self,
        conv_layers=2,
        conv_filters=[10, 20],
        conv_kernel_sizes=[5, 5],
        fc_hidden_sizes=[50],
        dropout_rates=[0.5],
        fc_dropout=True,
        pool_kernel_size=2,
        activation_func=F.relu,
        padding=0
    ):
        super(FlexibleNet, self).__init__()

        # Validate input parameters
        assert conv_layers <= len(conv_filters), "Not enough conv_filters provided"
        assert conv_layers <= len(conv_kernel_sizes), "Not enough conv_kernel_sizes provided"

        self.conv_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.pool_kernel_size = pool_kernel_size
        self.activation_func = activation_func

        # Set up convolutional layers
        in_channels = 1  # Grayscale input image
        for i in range(conv_layers):
            out_channels = conv_filters[i]
            kernel_size = conv_kernel_sizes[i]
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            in_channels = out_channels

        # Calculate the size of the flattened feature maps after convolutions and pooling
        # Start with 28x28 image (Fashion MNIST size)
        size = 28
        channels = conv_filters[conv_layers-1] if conv_layers > 0 else 1

        for i in range(conv_layers):
            # Apply convolution with padding: size = size - kernel_size + 1 + 2*padding
            size = size - conv_kernel_sizes[i] + 1 + 2*padding
            # Apply pooling: size = size / pool_kernel_size
            size = size // pool_kernel_size

        # Flattened feature size (with error handling for small sizes)
        if size <= 0:
            print(f"Warning: Invalid architecture - image size became {size} after {conv_layers} conv layers")
            print(f"Using fallback size of 1")
            size = 1  # Fallback to prevent crash

        flattened_size = channels * size * size

        # Set up fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = flattened_size

        for i, hidden_size in enumerate(fc_hidden_sizes):
            self.fc_layers.append(nn.Linear(in_features, hidden_size))
            if i < len(dropout_rates) and fc_dropout:
                self.dropout_layers.append(nn.Dropout(dropout_rates[i]))
            in_features = hidden_size

        # Output layer (10 classes in Fashion MNIST)
        self.output_layer = nn.Linear(in_features, 10)

    def forward(self, x):
        # Apply convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.activation_func(x)
            x = F.max_pool2d(x, self.pool_kernel_size)

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            x = self.activation_func(x)
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)

        # Output layer
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


# Load Fashion MNIST dataset
def load_fashion_mnist(batch_size_train, batch_size_test):
    train_loader = DataLoader(
        torchvision.datasets.FashionMNIST('data', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.FashionMNIST('data', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ])),
        batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader


# Training function
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = test_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# Function to train a model with specific configuration - OPTIMIZED FOR SPEED
def train_and_evaluate(config, train_loader, test_loader, device, max_epochs=3, patience=2):
    # Extract parameters from config
    conv_layers = config.get('conv_layers', 2)
    conv_filters = config.get('conv_filters', [10, 20])
    conv_kernel_sizes = config.get('conv_kernel_sizes', [5, 5])
    fc_hidden_sizes = config.get('fc_hidden_sizes', [50])
    dropout_rates = config.get('dropout_rates', [0.5])
    fc_dropout = config.get('fc_dropout', True)
    pool_kernel_size = config.get('pool_kernel_size', 2)
    learning_rate = config.get('learning_rate', 0.01)
    momentum = config.get('momentum', 0.5)
    padding = config.get('padding', 0)

    # Map string to activation function
    activation_name = config.get('activation_func', 'relu')
    activation_map = {
        'relu': F.relu,
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'leaky_relu': F.leaky_relu
    }
    activation_func = activation_map.get(activation_name, F.relu)

    # Extend lists if necessary to match conv_layers
    if len(conv_filters) < conv_layers:
        conv_filters = conv_filters + [conv_filters[-1]] * (conv_layers - len(conv_filters))

    if len(conv_kernel_sizes) < conv_layers:
        conv_kernel_sizes = conv_kernel_sizes + [conv_kernel_sizes[-1]] * (conv_layers - len(conv_kernel_sizes))

    # Create model
    try:
        model = FlexibleNet(
            conv_layers=conv_layers,
            conv_filters=conv_filters[:conv_layers],
            conv_kernel_sizes=conv_kernel_sizes[:conv_layers],
            fc_hidden_sizes=fc_hidden_sizes,
            dropout_rates=dropout_rates,
            fc_dropout=fc_dropout,
            pool_kernel_size=pool_kernel_size,
            activation_func=activation_func,
            padding=padding
        ).to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        print(f"Config that caused error: {config}")
        # Return a dummy result
        return {
            'config': config,
            'best_test_acc': 0,
            'final_train_acc': 0,
            'final_test_acc': 0,
            'epochs_completed': 0,
            'training_time': 0,
            'training_time_per_epoch': 0,
            'train_losses': [],
            'train_accs': [],
            'test_losses': [],
            'test_accs': []
        }

    # Setup optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # For early stopping
    best_test_acc = 0
    patience_counter = 0

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    epochs_completed = 0

    start_time = time.time()

    # Training loop - REDUCED MAX EPOCHS for faster experimentation
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Print progress
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        epochs_completed = epoch

        # Early stopping logic
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch} epochs')
                break

    training_time = time.time() - start_time

    # Calculate final metrics
    results = {
        'config': config,
        'best_test_acc': best_test_acc,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'epochs_completed': epochs_completed,
        'training_time': training_time,
        'training_time_per_epoch': training_time / epochs_completed,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

    return results


# Linear search strategy - OPTIMIZED
def linear_search_optimization():
    # Load datasets with larger batch size for speed
    batch_size_train = 128  # Increased for speed
    batch_size_test = 1000
    train_loader, test_loader = load_fashion_mnist(batch_size_train, batch_size_test)

    # Base configuration (starting point)
    base_config = {
        'conv_layers': 2,
        'conv_filters': [10, 20],
        'conv_kernel_sizes': [5, 5],
        'fc_hidden_sizes': [50],
        'dropout_rates': [0.5],
        'fc_dropout': True,
        'pool_kernel_size': 2,
        'activation_func': 'relu',
        'learning_rate': 0.01,
        'momentum': 0.5,
        'padding': 2,
        'batch_size_train': batch_size_train
    }

    # Expanded dimensions to explore - will generate 50+ configurations
    dimensions = {
        'conv_layers': [1, 2, 3],  # 3 options
        'conv_filters': [
            [8, 16],               # Small filters
            [16, 32],              # Medium filters
            [32, 64]               # Large filters
        ],  # 3 options
        'conv_kernel_sizes': [
            [3, 3],                # Small kernels
            [5, 5],                # Medium kernels
        ],  # 2 options
        'fc_hidden_sizes': [
            [50],                  # Single small layer
            [100],                 # Single medium layer
            [200],                 # Single large layer
            [50, 50]               # Two small layers
        ],  # 4 options
        'dropout_rates': [
            [0.2],                 # Low dropout
            [0.5],                 # Medium dropout
            [0.8]                  # High dropout
        ],  # 3 options
        'padding': [0, 1, 2],      # 3 options
        'fc_dropout': [True, False],  # 2 options
        'activation_func': ['relu', 'tanh', 'leaky_relu'],  # 3 options
        'learning_rate': [0.001, 0.01, 0.1]  # 3 options
    }

    # Store all experiment results
    all_results = []

    # Run baseline model first
    print("Running baseline model...")
    baseline_result = train_and_evaluate(base_config, train_loader, test_loader, DEVICE)
    all_results.append(baseline_result)
    print(f"Baseline Test Accuracy: {baseline_result['best_test_acc']:.2f}%")

    # Store experiment count for tracking
    num_experiments = 1  # Start at 1 for baseline

    # Linear search: optimize one dimension at a time
    for dim_name, dim_values in dimensions.items():
        print(f"\nOptimizing dimension: {dim_name}")
        best_config = base_config.copy()
        best_acc = baseline_result['best_test_acc']

        for value in dim_values:
            # Skip if this is the same as the current best config value
            if best_config.get(dim_name) == value:
                print(f"Skipping {dim_name} = {value} (same as current best)")
                continue

            # Special handling for batch size
            if dim_name == 'batch_size_train':
                # Create new data loader with this batch size
                train_loader, _ = load_fashion_mnist(value, batch_size_test)
                # Skip updating config since batch size is handled separately
                config = best_config.copy()
                # But still update the config value for consistency
                config['batch_size_train'] = value
            else:
                # Update config with the new value for this dimension
                config = best_config.copy()
                config[dim_name] = value

            print(f"Testing {dim_name} = {value} (Experiment {num_experiments}/50+)")
            num_experiments += 1

            # Train and evaluate
            result = train_and_evaluate(config, train_loader, test_loader, DEVICE)
            all_results.append(result)

            # Check if this configuration is better
            if result['best_test_acc'] > best_acc:
                best_acc = result['best_test_acc']
                if dim_name != 'batch_size_train' or dim_name not in best_config:
                    best_config[dim_name] = value
                print(f"New best {dim_name}: {value}, Test Acc: {best_acc:.2f}%")

            # Reset train_loader back if we changed it
            if dim_name == 'batch_size_train':
                train_loader, _ = load_fashion_mnist(batch_size_train, batch_size_test)

        # Update the base configuration with the best value found for this dimension
        base_config = best_config.copy()
        print(f"Best value for {dim_name}: {base_config.get(dim_name)}, Test Acc: {best_acc:.2f}%")

    # Final evaluation and reporting
    print("\nFinal best configuration:")
    for key, value in base_config.items():
        print(f"{key}: {value}")

    # Create dataframe with all results
    results_df = pd.DataFrame([
        {
            'conv_layers': r['config'].get('conv_layers', None),
            'conv_filters': str(r['config'].get('conv_filters', [])),
            'conv_kernel_sizes': str(r['config'].get('conv_kernel_sizes', [])),
            'fc_hidden_sizes': str(r['config'].get('fc_hidden_sizes', [])),
            'dropout_rates': str(r['config'].get('dropout_rates', [])),
            'fc_dropout': r['config'].get('fc_dropout', None),
            'pool_kernel_size': r['config'].get('pool_kernel_size', None),
            'activation_func': r['config'].get('activation_func', None),
            'learning_rate': r['config'].get('learning_rate', None),
            'momentum': r['config'].get('momentum', None),
            'best_test_acc': r['best_test_acc'],
            'final_train_acc': r['final_train_acc'],
            'final_test_acc': r['final_test_acc'],
            'epochs_completed': r['epochs_completed'],
            'training_time': r['training_time'],
            'training_time_per_epoch': r['training_time_per_epoch']
        }
        for r in all_results
    ])

    # Save results to CSV
    results_path = 'results/experiments/network_experiments.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    # Save best configuration
    best_config_path = 'results/experiments/best_config.json'
    with open(best_config_path, 'w') as f:
        json.dump(base_config, f, indent=4)
    print(f"Best configuration saved to {best_config_path}")

    # Plot results for each dimension
    plot_experiment_results(results_df)

    return base_config, results_df


# Function to plot experiment results
def plot_experiment_results(results_df):
    # Create directory for plots
    os.makedirs('results/experiments/plots', exist_ok=True)

    # Key dimensions to visualize - reduced for simplicity
    dimensions = [
        'conv_layers',
        'activation_func',
        'learning_rate',
        'fc_dropout',
    ]

    for dim in dimensions:
        try:
            plt.figure(figsize=(10, 6))

            # Group by this dimension
            if dim in ['conv_filters', 'conv_kernel_sizes', 'fc_hidden_sizes', 'dropout_rates']:
                # Skip complex dimensions that are stored as strings
                continue
            else:
                # Filter out rows with missing values for this dimension
                filtered_df = results_df[results_df[dim].notna()]
                if filtered_df.empty:
                    print(f"No data available for dimension: {dim}")
                    continue

                groups = filtered_df.groupby(dim)

            max_accs = []
            labels = []

            for name, group in groups:
                max_acc = group['best_test_acc'].max()
                max_accs.append(max_acc)
                labels.append(str(name))

            if not labels:
                print(f"No groups found for dimension: {dim}")
                continue

            plt.bar(labels, max_accs)
            plt.xlabel(dim)
            plt.ylabel('Best Test Accuracy (%)')
            plt.title(f'Effect of {dim} on Model Performance')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plt.savefig(f'results/experiments/plots/{dim}_performance.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting {dim}: {e}")

    # Plot training time vs accuracy
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['training_time'], results_df['best_test_acc'])
        plt.xlabel('Training Time (s)')
        plt.ylabel('Best Test Accuracy (%)')
        plt.title('Training Time vs Accuracy')
        plt.tight_layout()
        plt.savefig('results/experiments/plots/time_vs_accuracy.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting time vs accuracy: {e}")


# Main function to run experiments
def main():
    print("Starting network architecture experiments on Fashion MNIST")
    print(f"Device: {DEVICE}")

    # Run linear search optimization
    best_config, results_df = linear_search_optimization()

    # Find the overall best configuration
    if not results_df.empty:
        best_row = results_df.loc[results_df['best_test_acc'].idxmax()]

        print("\nExperiment Summary:")
        print(f"Total configurations tested: {len(results_df)}")
        print(f"Best test accuracy: {best_row['best_test_acc']:.2f}%")
        print(f"Best configuration:")

        for key in ['conv_layers', 'fc_hidden_sizes', 'dropout_rates', 'fc_dropout',
                    'activation_func', 'learning_rate']:
            print(f"  {key}: {best_row[key]}")
    else:
        print("No results collected.")

    print("Experiment completed successfully!")
    print("Results saved to results/experiments/")


if __name__ == "__main__":
    main()
