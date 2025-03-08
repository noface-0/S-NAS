import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Dict, List, Any, Tuple, Optional, Union

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
                 activation='relu', use_batch_norm=False, dropout_rate=0.0,
                 stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # Calculate padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        # First conv layer (with stride)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Second conv layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
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
        
        # Downsample option for residual path
        self.downsample = downsample
        
        # Skip connection with 1x1 conv if dimensions don't match
        self.use_shortcut = in_channels != out_channels and downsample is None
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
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
        
        # Apply downsample path if provided
        if self.downsample is not None:
            identity = self.downsample(x)
        # Apply shortcut connection if needed
        elif self.use_shortcut:
            identity = self.shortcut(x)
        
        # Add skip connection (with shape check)
        if out.shape == identity.shape:
            out = out + identity
        else:
            # If shapes don't match, use 1x1 conv to adjust identity dimensions
            if not hasattr(self, 'shape_adjuster'):
                self.shape_adjuster = nn.Conv2d(
                    in_channels=identity.size(1),
                    out_channels=out.size(1),
                    kernel_size=1,
                    stride=1 if out.shape[2] == identity.shape[2] else 2
                ).to(out.device)
            identity_adjusted = self.shape_adjuster(identity)
            out = out + identity_adjusted
            
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
        
        # Apply width multiplier (if not already applied) with minimum size
        in_channels_adj = max(8, int(in_channels * width_multiplier))
        out_channels_adj = max(8, int(out_channels * width_multiplier))
        
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
        # Ensure skip connections is a list
        skip_connections = architecture.get('use_skip_connections', [False] * num_layers)
        if not isinstance(skip_connections, list):
            skip_connections = [skip_connections] * num_layers
        use_skip_connections = skip_connections
        
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
                    # Check if shapes match completely for direct addition
                    if skip_output.shape == x.shape:
                        # Add skip connection
                        x = x + skip_output
                        break
                    # For mismatched dimensions, we'll skip rather than adapting to avoid instability
                    # This is safer than attempting to adapt dimensions during forward pass
            
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
            out_channels_raw = filters[i]
            # Apply width multiplier here
            out_channels_adj = max(int(out_channels_raw * width_multiplier), 8)
            kernel_size = kernel_sizes[i]
            activation = activations[i]
            
            # Create depthwise separable block with pre-adjusted channel counts
            # and width_multiplier=1.0 to prevent double application
            ds_block = DepthwiseSeparableBlock(
                in_channels=in_channels,
                out_channels=out_channels_adj,
                kernel_size=kernel_size,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
                width_multiplier=1.0  # Set to 1.0 to prevent double application
            )
            
            self.ds_blocks.append(ds_block)
            in_channels = out_channels_adj  # Use already adjusted value for next layer
        
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

# DenseNet Implementation
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, activation='relu'):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # Bottleneck layer to reduce computation
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, inputs):
        x = torch.cat(inputs, 1)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        
        return x


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, activation='relu'):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                activation=activation
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, activation='relu'):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        
        # Activation function
        if activation == 'relu':
            self.add_module('relu', nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            self.add_module('relu', nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'elu':
            self.add_module('relu', nn.ELU(inplace=True))
        elif activation == 'selu':
            self.add_module('relu', nn.SELU(inplace=True))
        else:
            self.add_module('relu', nn.ReLU(inplace=True))
            
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNetModel(nn.Module):
    """DenseNet model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the DenseNet model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(DenseNetModel, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        growth_rate = architecture.get('growth_rate', 32)
        block_config = architecture.get('block_config', [6, 12, 24, 16])
        num_init_features = architecture.get('num_init_features', 64)
        bn_size = architecture.get('bn_size', 4)
        compression_factor = architecture.get('compression_factor', 0.5)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        activation = architecture.get('activation', 'relu')
        
        # First convolution
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(input_shape[0], num_init_features, kernel_size=7, 
                                                   stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        
        # Activation
        if activation == 'relu':
            self.features.add_module('relu0', nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            self.features.add_module('relu0', nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'elu':
            self.features.add_module('relu0', nn.ELU(inplace=True))
        elif activation == 'selu':
            self.features.add_module('relu0', nn.SELU(inplace=True))
        else:
            self.features.add_module('relu0', nn.ReLU(inplace=True))
        
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Add a dense block
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=dropout_rate,
                activation=activation
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            # Add a transition layer after each dense block (except the last one)
            if i != len(block_config) - 1:
                out_features = int(num_features * compression_factor)
                trans = _Transition(num_features, out_features, activation=activation)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = out_features
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        # Final activation
        if activation == 'relu':
            self.features.add_module('relu_final', nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            self.features.add_module('relu_final', nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'elu':
            self.features.add_module('relu_final', nn.ELU(inplace=True))
        elif activation == 'selu':
            self.features.add_module('relu_final', nn.SELU(inplace=True))
        else:
            self.features.add_module('relu_final', nn.ReLU(inplace=True))
        
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# ShuffleNetV2 Implementation
class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        # Channel shuffle: [N, C, H, W] -> [N, g, C//g, H, W] -> [N, C//g, g, H, W] -> [N, C, H, W]
        N, C, H, W = x.size()
        g = self.groups
        # Handle the case where C is not divisible by g
        if C % g != 0:
            # Ensure we have a proper division
            g = math.gcd(C, g)  # Find greatest common divisor
            if g <= 1:  # Fallback if no common divisor
                return x  # Skip shuffling
        return x.view(N, g, C//g, H, W).transpose(1, 2).contiguous().view(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio=0.5):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='relu'):
        super(ShuffleNetV2Block, self).__init__()
        self.stride = stride
        
        out_channels = out_channels // 2
        
        # Determine activation function
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.act = nn.SELU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
        
        if stride == 2:
            # Downsampling path
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.act
            )
            
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.act,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.act
            )
        else:
            # Equal split for normal blocks
            assert in_channels % 2 == 0
            self.branch1 = nn.Sequential()  # Identity for the branch1
            in_channels = in_channels // 2
            
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                self.act,
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                self.act
            )
        
        self.shuffle = ShuffleBlock()
    
    def forward(self, x):
        if self.stride == 1:
            # Split the channel
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # No split for downsampling
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        
        out = self.shuffle(out)
        return out


class ShuffleNetV2Model(nn.Module):
    """ShuffleNetV2 model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the ShuffleNetV2 model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(ShuffleNetV2Model, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        width_multiplier = architecture.get('width_multiplier', 1.0)
        out_channels = architecture.get('out_channels', [116, 232, 464, 1024])
        num_blocks_per_stage = architecture.get('num_blocks_per_stage', [4, 8, 4])
        activation = architecture.get('activation', 'relu')
        
        # Scale channels by width_multiplier
        out_channels = [int(c * width_multiplier) for c in out_channels]
        
        # Ensure channels are divisible by 2
        out_channels = [c + (c % 2) for c in out_channels]
        
        in_channels = 24  # Initial convolution output channels
        
        # First conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Building stages
        self.stages = nn.Sequential()
        for i, num_blocks in enumerate(num_blocks_per_stage):
            stage_out_channels = out_channels[i]
            
            # First block of each stage with stride=2
            self.stages.add_module(f'stage{i+1}_block1', 
                                  ShuffleNetV2Block(in_channels, stage_out_channels, stride=2, activation=activation))
            
            # Rest of blocks in the stage with stride=1
            for j in range(1, num_blocks):
                self.stages.add_module(f'stage{i+1}_block{j+1}', 
                                      ShuffleNetV2Block(stage_out_channels, stage_out_channels, stride=1, activation=activation))
            
            in_channels = stage_out_channels
        
        # Final 1x1 conv
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels[-1], num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Enhanced MLPModel with residual connections and layer normalization
class EnhancedMLPModel(nn.Module):
    """Enhanced MLP model with residual connections and layer normalization."""
    
    def __init__(self, architecture):
        """
        Initialize the Enhanced MLP model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(EnhancedMLPModel, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        num_layers = architecture['num_layers']
        hidden_units = architecture['hidden_units']
        activations = architecture['activations']
        dropout_rate = architecture.get('dropout_rate', 0.0)
        use_residual = architecture.get('use_residual', False)
        use_layer_norm = architecture.get('use_layer_norm', False)
        
        # Calculate input size from input shape
        input_size = input_shape[0] * input_shape[1] * input_shape[2]  # Flatten image
        
        # Create fully-connected layers
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        
        in_features = input_size
        for i in range(num_layers):
            # Create layer block with residual connection if specified
            layer_block = nn.ModuleDict()
            
            # Linear transformation
            layer_block['linear'] = nn.Linear(in_features, hidden_units[i])
            
            # Layer normalization
            if use_layer_norm:
                layer_block['norm'] = nn.LayerNorm(hidden_units[i])
            
            # Activation function
            if activations[i] == 'relu':
                layer_block['activation'] = nn.ReLU(inplace=True)
            elif activations[i] == 'leaky_relu':
                layer_block['activation'] = nn.LeakyReLU(0.1, inplace=True)
            elif activations[i] == 'elu':
                layer_block['activation'] = nn.ELU(inplace=True)
            elif activations[i] == 'selu':
                layer_block['activation'] = nn.SELU(inplace=True)
            elif activations[i] == 'gelu':
                layer_block['activation'] = nn.GELU()
            else:
                layer_block['activation'] = nn.ReLU(inplace=True)
            
            # Dropout
            if dropout_rate > 0:
                layer_block['dropout'] = nn.Dropout(dropout_rate)
            
            # Track if residual connection is possible
            can_use_residual = use_residual and i > 0 and in_features == hidden_units[i]
            layer_block['use_residual'] = can_use_residual
            
            self.layers.append(layer_block)
            in_features = hidden_units[i]
        
        # Output layer
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # Flatten the input
        x = self.flatten(x)
        
        # Process through layers
        for layer in self.layers:
            identity = x  # Save for potential residual connection
            
            # Apply linear transformation
            x = layer['linear'](x)
            
            # Apply layer normalization if present
            if 'norm' in layer:
                x = layer['norm'](x)
            
            # Apply activation
            x = layer['activation'](x)
            
            # Apply dropout if present
            if 'dropout' in layer:
                x = layer['dropout'](x)
            
            # Apply residual connection if possible
            if layer.get('use_residual', False):
                x = x + identity
        
        # Classification layer
        x = self.classifier(x)
        
        return x


# EfficientNet Implementation (Simplified Version)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        squeeze_channels = max(1, int(channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, squeeze_channels, kernel_size=1),
            Swish(),
            nn.Conv2d(squeeze_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv Block with SE."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio=0.25, dropout_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        else:
            self.expand = nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, 
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )
        
        # Squeeze and Excitation
        self.se = SEBlock(expanded_channels, se_ratio)
        
        # Output projection
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout
        self.dropout_rate = dropout_rate
        if dropout_rate > 0 and self.use_residual:
            self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand(x)
        
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        # Projection
        x = self.project(x)
        
        # Skip connection
        if self.use_residual:
            if self.dropout_rate > 0:
                x = self.dropout(x)
            x = x + identity
        
        return x


class EfficientNetModel(nn.Module):
    """EfficientNet model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the EfficientNet model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(EfficientNetModel, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        width_factor = architecture.get('width_factor', 1.0)
        depth_factor = architecture.get('depth_factor', 1.0)
        dropout_rate = architecture.get('dropout_rate', 0.2)
        se_ratio = architecture.get('se_ratio', 0.25)
        
        # Base parameters
        base_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        base_channels = [int(c * width_factor) for c in base_channels]
        
        # MBConv block parameters: [expanded_ratio, kernel_size, stride, repeats]
        block_params = [
            [1, 3, 1, 1],  # MBConv1_3x3, stride 1, repeats 1
            [6, 3, 2, 2],  # MBConv6_3x3, stride 2, repeats 2
            [6, 5, 2, 2],  # MBConv6_5x5, stride 2, repeats 2
            [6, 3, 2, 3],  # MBConv6_3x3, stride 2, repeats 3
            [6, 5, 1, 3],  # MBConv6_5x5, stride 1, repeats 3
            [6, 5, 2, 4],  # MBConv6_5x5, stride 2, repeats 4
            [6, 3, 1, 1]   # MBConv6_3x3, stride 1, repeats 1
        ]
        
        # Adjust repeats based on depth factor
        block_repeats = [max(1, int(repeat * depth_factor)) for _, _, _, repeat in block_params]
        
        # Initial convolutional layer
        self.conv_stem = nn.Sequential(
            nn.Conv2d(input_shape[0], base_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[0]),
            Swish()
        )
        
        # Building blocks
        self.blocks = nn.Sequential()
        in_channels = base_channels[0]
        block_id = 0
        
        for stage_idx, (expand_ratio, kernel_size, stride, _) in enumerate(block_params):
            out_channels = base_channels[stage_idx + 1]
            repeats = block_repeats[stage_idx]
            
            # First block of each stage with the specified stride
            self.blocks.add_module(f'block{block_id}', 
                                  MBConvBlock(in_channels, out_channels, kernel_size, stride, 
                                             expand_ratio, se_ratio, dropout_rate))
            block_id += 1
            in_channels = out_channels
            
            # Rest of blocks in the stage with stride 1
            for _ in range(1, repeats):
                self.blocks.add_module(f'block{block_id}', 
                                      MBConvBlock(in_channels, out_channels, kernel_size, 1, 
                                                 expand_ratio, se_ratio, dropout_rate))
                block_id += 1
        
        # Final convolutional layer
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels[-1]),
            Swish()
        )
        
        # Global pooling and final linear layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(base_channels[-1], num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Stem
        x = self.conv_stem(x)
        
        # Blocks
        x = self.blocks(x)
        
        # Head
        x = self.conv_head(x)
        
        # Pooling and final linear layer
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class ResNetModel(nn.Module):
    """ResNet model built from an architecture configuration."""
    
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
        use_batch_norm = architecture.get('use_batch_norm', True)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_shape[0], filters[0], kernel_size=7, stride=2, padding=3, bias=False)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(filters[0])
        
        # First activation
        if activations[0] == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activations[0] == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activations[0] == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activations[0] == 'selu':
            self.activation = nn.SELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)  # Default to ReLU
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layers = nn.ModuleList()
        in_channels = filters[0]
        for i in range(num_layers - 1):  # -1 because we already have one layer
            out_channels = filters[min(i+1, len(filters)-1)]
            kernel_size = kernel_sizes[min(i+1, len(kernel_sizes)-1)]
            activation = activations[min(i+1, len(activations)-1)]
            
            # Create downsample layer if dimensions change
            downsample = None
            stride = 1
            if in_channels != out_channels:
                stride = 2  # Downsample when channels change
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
                )
            
            # Add residual block
            block = ResidualBlock(
                in_channels, 
                out_channels, 
                kernel_size, 
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
                stride=stride,
                downsample=downsample
            )
            
            self.layers.append(block)
            in_channels = out_channels
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        
        # Use dropout if specified
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        # Residual blocks
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout if needed
        if self.dropout_rate > 0:
            x = self.dropout(x)
        
        # Classification
        x = self.fc(x)
        
        return x


class MobileNetModel(nn.Module):
    """MobileNet model built from an architecture configuration."""
    
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
        use_batch_norm = architecture.get('use_batch_norm', True)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        width_multiplier = architecture.get('width_multiplier', 1.0)
        
        # Initial convolutional layer
        initial_filters = max(8, int(32 * width_multiplier))
        self.conv1 = nn.Conv2d(input_shape[0], initial_filters, kernel_size=3, stride=2, padding=1, bias=False)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(initial_filters)
        
        # First activation
        if activations[0] == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activations[0] == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activations[0] == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activations[0] == 'selu':
            self.activation = nn.SELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)  # Default to ReLU
        
        # Depthwise separable convolutional blocks
        self.layers = nn.ModuleList()
        in_channels = initial_filters
        
        for i in range(num_layers):
            out_channels = max(8, int(filters[min(i, len(filters)-1)] * width_multiplier))
            kernel_size = kernel_sizes[min(i, len(kernel_sizes)-1)]
            activation = activations[min(i, len(activations)-1)]
            
            # Stride = 2 for some layers to reduce spatial dimensions
            stride = 2 if i in [1, 3, 5, 11] and i < len(filters) else 1
            
            # Add depthwise separable block
            block = DepthwiseSeparableBlock(
                in_channels, 
                out_channels, 
                kernel_size, 
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
                width_multiplier=width_multiplier
            )
            
            self.layers.append(block)
            in_channels = out_channels
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        
        # Use dropout if specified
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.activation(x)
        
        # Depthwise separable blocks
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout if needed
        if self.dropout_rate > 0:
            x = self.dropout(x)
        
        # Classification
        x = self.fc(x)
        
        return x


class DenseNetModel(nn.Module):
    """DenseNet model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the DenseNet model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(DenseNetModel, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        growth_rate = architecture.get('growth_rate', 32)
        block_config = architecture.get('block_config', [6, 12, 24, 16])
        compression_factor = architecture.get('compression_factor', 0.5)
        bn_size = architecture.get('bn_size', 4)
        use_batch_norm = architecture.get('use_batch_norm', True)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        
        # Initial convolution
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(input_shape[0], 2 * growth_rate, 
                                                   kernel_size=7, stride=2, padding=3, bias=False))
        if use_batch_norm:
            self.features.add_module('norm0', nn.BatchNorm2d(2 * growth_rate))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # Each dense block and transition layer
        num_features = 2 * growth_rate
        for i, num_layers in enumerate(block_config):
            # Add a dense block
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=dropout_rate,
                activation='relu'
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # Add a transition layer after each dense block except the last one
            if i != len(block_config) - 1:
                num_output_features = int(num_features * compression_factor)
                trans = _Transition(num_features, num_output_features, 'relu')
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_output_features
        
        # Final batch norm
        if use_batch_norm:
            self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        
        # Global average pooling and classifier
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Use dropout if specified
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        
        if self.dropout_rate > 0:
            out = self.dropout(out)
            
        out = self.classifier(out)
        return out


class ShuffleNetV2Model(nn.Module):
    """ShuffleNetV2 model built from an architecture configuration."""
    
    def __init__(self, architecture):
        """
        Initialize the ShuffleNetV2 model based on the architecture specification.
        
        Args:
            architecture: Dictionary specifying the architecture
        """
        super(ShuffleNetV2Model, self).__init__()
        self.architecture = architecture
        
        # Extract architecture parameters
        input_shape = architecture['input_shape']
        num_classes = architecture['num_classes']
        width_multiplier = architecture.get('width_multiplier', 1.0)
        out_channels = architecture.get('out_channels', [116, 232, 464, 1024])
        num_blocks_per_stage = architecture.get('num_blocks_per_stage', [4, 8, 4])
        use_batch_norm = architecture.get('use_batch_norm', True)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        
        input_channels = input_shape[0]
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=True)
        
        # MaxPool
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build stages
        self.stages = nn.ModuleList()
        input_channels = 24
        
        for i, (num_blocks, output_channels) in enumerate(zip(num_blocks_per_stage, out_channels[:3])):
            # Apply width multiplier
            output_channels = int(output_channels * width_multiplier)
            
            # First block has stride 2
            stage = nn.Sequential()
            stage.add_module(f'block0', ShuffleNetV2Block(input_channels, output_channels, stride=2, activation='relu'))
            
            # Rest of blocks have stride 1
            for j in range(1, num_blocks):
                stage.add_module(f'block{j}', ShuffleNetV2Block(output_channels, output_channels, stride=1, activation='relu'))
                
            self.stages.append(stage)
            input_channels = output_channels
        
        # Final convolution
        self.conv5 = nn.Conv2d(input_channels, out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        if use_batch_norm:
            self.bn5 = nn.BatchNorm2d(out_channels[3])
        self.relu5 = nn.ReLU(inplace=True)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(out_channels[3], num_classes)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Final convolution
        x = self.conv5(x)
        if hasattr(self, 'bn5'):
            x = self.bn5(x)
        x = self.relu5(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
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
        elif network_type == 'enhanced_mlp':
            self.model = EnhancedMLPModel(architecture)
        elif network_type == 'resnet':
            self.model = ResNetModel(architecture)
        elif network_type == 'mobilenet':
            self.model = MobileNetModel(architecture)
        elif network_type == 'densenet':
            self.model = DenseNetModel(architecture)
        elif network_type == 'shufflenetv2':
            self.model = ShuffleNetV2Model(architecture)
        elif network_type == 'efficientnet':
            self.model = EfficientNetModel(architecture)
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
        
    def wrap_model_for_distributed(self, model, device, ddp_rank=None, ddp_world_size=None, 
                                  use_sync_bn=True, gradient_accumulation_steps=1):
        """
        Wrap a model for distributed training using DDPModel.
        
        Args:
            model: The base model to wrap
            device: The device to use
            ddp_rank: The rank in distributed training
            ddp_world_size: The total number of processes
            use_sync_bn: Whether to use synchronized batch normalization
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            DDPModel: The wrapped model
        """
        # Import here to avoid circular imports
        from ..utils.job_distributor import DDPModel
        
        # Wrap the model with DDPModel
        wrapped_model = DDPModel(
            model,
            device,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            use_sync_bn=use_sync_bn,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        return wrapped_model