"""
Model Builder for S-NAS

This module converts architecture descriptions into executable PyTorch models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any, Tuple, Optional

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

class DenseBlock(nn.Module):
    """A fully-connected block with optional batch norm, activation, and dropout."""
    
    def __init__(self, in_features, out_features, 
                 activation='relu', use_batch_norm=False, dropout_rate=0.0):
        super(DenseBlock, self).__init__()
        
        # Linear layer
        self.fc = nn.Linear(in_features, out_features)
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_features)
        
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
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.fc(x)
        
        if self.use_batch_norm:
            x = self.bn(x)
        
        x = self.activation(x)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
        
        return x

class ResidualBlock(nn.Module):
    """A residual block with skip connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 activation='relu', use_batch_norm=False, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        
        # Calculate padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        # First conv layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Second conv layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
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
        
        # Skip connection with 1x1 conv if dimensions don't match
        self.use_shortcut = in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0
            )
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        
        # Second conv block
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        # Apply shortcut connection if needed
        if self.use_shortcut:
            identity = self.shortcut(x)
        
        # Add skip connection
        out += identity
        out = self.activation(out)
        
        if self.dropout_rate > 0:
            out = self.dropout(out)
        
        return out

class DepthwiseSeparableBlock(nn.Module):
    """A depthwise separable convolution block (MobileNet style)."""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 activation='relu', use_batch_norm=False, dropout_rate=0.0,
                 width_multiplier=1.0):
        super(DepthwiseSeparableBlock, self).__init__()
        
        # Apply width multiplier
        in_channels_adj = max(1, int(in_channels * width_multiplier))
        out_channels_adj = max(1, int(out_channels * width_multiplier))
        
        # Calculate padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels_adj,
            out_channels=in_channels_adj,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels_adj  # Each input channel is convolved separately
        )
        
        # Pointwise convolution (1x1 conv)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels_adj,
            out_channels=out_channels_adj,
            kernel_size=1
        )
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn_depthwise = nn.BatchNorm2d(in_channels_adj)
            self.bn_pointwise = nn.BatchNorm2d(out_channels_adj)
        
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
        # Depthwise convolution
        x = self.depthwise(x)
        if self.use_batch_norm:
            x = self.bn_depthwise(x)
        x = self.activation(x)
        
        # Pointwise convolution
        x = self.pointwise(x)
        if self.use_batch_norm:
            x = self.bn_pointwise(x)
        x = self.activation(x)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
        
        return x

class MLPModel(nn.Module):
    """MLP model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the MLP model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(MLPModel, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        num_layers = architecture['num_layers']
        hidden_units = architecture['hidden_units']
        activations = architecture['activations']
        use_batch_norm = architecture.get('use_batch_norm', False)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        
        # Calculate input size from input shape
        input_size = input_shape[0] * input_shape[1] * input_shape[2]  # Flatten image
        
        # Create fully-connected layers
        self.fc_layers = nn.ModuleList()
        
        # Input layer
        in_features = input_size
        for i in range(num_layers):
            # Create dense block
            fc_block = DenseBlock(
                in_features=in_features,
                out_features=hidden_units[i],
                activation=activations[i],
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
            
            self.fc_layers.append(fc_block)
            in_features = hidden_units[i]  # Output becomes input for next layer
        
        # Output layer
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Forward through fully-connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Classification layer
        x = self.classifier(x)
        
        return x

class ConvolutionalModel(nn.Module):
    """Standard convolutional model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the convolutional model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(ConvolutionalModel, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
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
        
        # Classification layer
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

class ResNetModel(nn.Module):
    """ResNet-style model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the ResNet model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(ResNetModel, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        num_layers = architecture['num_layers']
        filters = architecture['filters']
        kernel_sizes = architecture['kernel_sizes']
        activations = architecture['activations']
        use_batch_norm = architecture.get('use_batch_norm', True)  # Default True for ResNet
        dropout_rate = architecture.get('dropout_rate', 0.0)
        
        # Input channels from the input shape
        in_channels = input_shape[0]  # First dimension is channels
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters[0],
            kernel_size=3,
            padding=1,
            stride=1
        )
        
        if use_batch_norm:
            self.initial_bn = nn.BatchNorm2d(filters[0])
        
        # Activation
        if activations[0] == 'relu':
            self.initial_activation = nn.ReLU(inplace=True)
        elif activations[0] == 'leaky_relu':
            self.initial_activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.initial_activation = nn.ReLU(inplace=True)
        
        # Create residual blocks
        self.res_blocks = nn.ModuleList()
        in_channels = filters[0]
        
        for i in range(num_layers):
            out_channels = filters[i]
            kernel_size = kernel_sizes[i]
            activation = activations[i]
            
            # Create residual block
            res_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
            
            self.res_blocks.append(res_block)
            in_channels = out_channels  # Output becomes input for next layer
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification layer
        self.classifier = nn.Linear(filters[-1], num_classes)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        if hasattr(self, 'initial_bn'):
            x = self.initial_bn(x)
            
        x = self.initial_activation(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Global average pooling (reduces to B x C x 1 x 1)
        x = self.global_avg_pool(x)
        
        # Flatten (B x C)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.classifier(x)
        
        return x

class MobileNetModel(nn.Module):
    """MobileNet-style model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the MobileNet model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(MobileNetModel, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        num_layers = architecture['num_layers']
        filters = architecture['filters']
        kernel_sizes = architecture['kernel_sizes']
        activations = architecture['activations']
        use_batch_norm = architecture.get('use_batch_norm', True)  # Default True for MobileNet
        dropout_rate = architecture.get('dropout_rate', 0.0)
        width_multiplier = architecture.get('width_multiplier', 1.0)
        
        # Input channels from the input shape
        in_channels = input_shape[0]  # First dimension is channels
        
        # Initial standard convolution
        out_channels = max(int(32 * width_multiplier), 8)
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=2  # Reduction in spatial dimensions
        )
        
        if use_batch_norm:
            self.initial_bn = nn.BatchNorm2d(out_channels)
        
        # Activation
        if activations[0] == 'relu':
            self.initial_activation = nn.ReLU(inplace=True)
        elif activations[0] == 'leaky_relu':
            self.initial_activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.initial_activation = nn.ReLU(inplace=True)
        
        # Create depthwise separable blocks
        self.ds_blocks = nn.ModuleList()
        in_channels = out_channels
        
        for i in range(num_layers):
            out_channels = filters[i]
            kernel_size = kernel_sizes[i]
            activation = activations[i]
            
            # Create depthwise separable block
            ds_block = DepthwiseSeparableBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
                width_multiplier=width_multiplier
            )
            
            self.ds_blocks.append(ds_block)
            in_channels = max(int(out_channels * width_multiplier), 8)  # Output becomes input for next layer
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification layer
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        if hasattr(self, 'initial_bn'):
            x = self.initial_bn(x)
            
        x = self.initial_activation(x)
        
        # Depthwise separable blocks
        for ds_block in self.ds_blocks:
            x = ds_block(x)
        
        # Global average pooling (reduces to B x C x 1 x 1)
        x = self.global_avg_pool(x)
        
        # Flatten (B x C)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.classifier(x)
        
        return x

class SearchModel(nn.Module):
    """Neural network model selector based on architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(SearchModel, self).__init__()
        
        # Create the appropriate model based on network_type
        network_type = architecture.get('network_type', 'cnn')
        
        if network_type == 'mlp':
            self.model = MLPModel(architecture)
        elif network_type == 'resnet':
            self.model = ResNetModel(architecture)
        elif network_type == 'mobilenet':
            self.model = MobileNetModel(architecture)
        else:  # Default to CNN
            self.model = ConvolutionalModel(architecture)
    
    def forward(self, x):
        return self.model(x)

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