"""
Model Builder for S-NAS

This module converts architecture descriptions into executable PyTorch models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any

class ConvBlock(nn.Module):
    """A convolutional block with optional batch norm, activation, and dropout."""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 activation='relu', use_batch_norm=False, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        
        # Calculate padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        # Conv layer
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)  # Default to ReLU
        
        # Dropout
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        x = self.conv(x)
        
        if self.use_batch_norm:
            x = self.bn(x)
        
        x = self.activation(x)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
        
        return x

class SearchModel(nn.Module):
    """Neural network model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(SearchModel, self).__init__()
        self.architecture = architecture
        
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        num_layers = architecture['num_layers']
        filters = architecture['filters']
        kernel_sizes = architecture['kernel_sizes']
        activations = architecture['activations']
        use_batch_norm = architecture.get('use_batch_norm', False)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        use_skip_connections = architecture.get('use_skip_connections', [False] * num_layers)
        
        # Input channels from the input shape
        in_channels = input_shape[0]  # First dimension is channels
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            out_channels = filters[i]
            kernel_size = kernel_sizes[i]
            activation = activations[i]
            
            # Create convolutional block
            conv_block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
            
            self.conv_layers.append(conv_block)
            in_channels = out_channels  # Output becomes input for next layer
        
        # Store skip connection configuration
        self.use_skip_connections = use_skip_connections
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Calculate the size of flattened features
        # The output size after the last conv layer
        self.classifier = nn.Linear(filters[-1], num_classes)
    
    def forward(self, x):
        # Save intermediate outputs for skip connections
        skip_outputs = []
        
        # Forward through convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            # Apply skip connection if specified and possible
            if i > 0 and self.use_skip_connections[i] and skip_outputs:
                # Get the most recent skip output with matching channels
                for skip_output in reversed(skip_outputs):
                    if skip_output.shape[1] == x.shape[1]:  # Check channel dimension
                        # Add skip connection
                        x = x + skip_output
                        break
            
            # Apply convolutional block
            x = conv_layer(x)
            
            # Save output for potential skip connections
            skip_outputs.append(x)
        
        # Global average pooling (reduces to B x C x 1 x 1)
        x = self.global_avg_pool(x)
        
        # Flatten (B x C)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.classifier(x)
        
        return x

class ModelBuilder:
    """Builds executable PyTorch models from architecture descriptions."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the model builder.
        
        Args:
            device: Device to place the model on ('cuda' or 'cpu')
        """
        self.device = device
    
    def build_model(self, architecture):
        """
        Build a PyTorch model from the architecture description.
        
        Args:
            architecture: Dictionary specifying the architecture
            
        Returns:
            nn.Module: PyTorch model
        """
        model = SearchModel(architecture)
        model = model.to(self.device)
        return model
    
    def build_optimizer(self, model, architecture):
        """
        Build an optimizer based on the architecture description.
        
        Args:
            model: PyTorch model
            architecture: Dictionary specifying the architecture
            
        Returns:
            torch.optim.Optimizer: PyTorch optimizer
        """
        optimizer_name = architecture.get('optimizer', 'adam')
        learning_rate = architecture.get('learning_rate', 0.001)
        
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
        else:
            # Default to Adam
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
        
        return optimizer
    
    def get_criterion(self, architecture):
        """
        Get a loss function for the architecture.
        
        Args:
            architecture: Dictionary specifying the architecture
            
        Returns:
            criterion: PyTorch loss function
        """
        # For classification tasks, we use cross-entropy loss
        return nn.CrossEntropyLoss()
    
    def build_training_components(self, architecture):
        """
        Build all components needed for training a model.
        
        Args:
            architecture: Dictionary specifying the architecture
            
        Returns:
            tuple: (model, optimizer, criterion)
        """
        model = self.build_model(architecture)
        optimizer = self.build_optimizer(model, architecture)
        criterion = self.get_criterion(architecture)
        
        return model, optimizer, criterion