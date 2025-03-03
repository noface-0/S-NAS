import random
import copy
from typing import Dict, List, Any

class ArchitectureSpace:
    """Defines the space of possible neural network architectures."""
    
    def __init__(self, input_shape, num_classes):
        """
        Initialize the architecture space.
        
        Args:
            input_shape: Input shape of the data (channels, height, width)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Define the possible values for each architecture parameter
        self.space = {
            # Network type
            'network_type': ['cnn', 'mlp', 'enhanced_mlp', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet'],
            
            # Number of layers in the network
            'num_layers': [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 50],
            
            # Possible number of filters (neurons) for each layer
            'filters': [16, 32, 64, 128, 256, 512, 1024],
            
            # Possible kernel sizes for conv layers
            'kernel_sizes': [3, 5, 7, 9],
            
            # Possible hidden units for dense layers (MLP)
            'hidden_units': [64, 128, 256, 512, 1024, 2048, 4096],
            
            # Possible activation functions
            'activations': ['relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
            
            # Possible learning rates
            'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],
            
            # Possible batch normalization options
            'use_batch_norm': [True, False],
            
            # Possible dropout options
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            
            # Possible skip connection option values (each layer gets one of these) 
            'use_skip_connections': [True, False],
            
            # Possible growth factors for mobilenet
            'width_multiplier': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            
            # Possible optimizers
            'optimizer': ['sgd', 'adam', 'adamw', 'rmsprop'],
        }
    
    def get_options(self, parameter_name):
        """
        Get possible values for a specific parameter.
        
        Args:
            parameter_name: Name of the architecture parameter
            
        Returns:
            list: Possible values for the parameter
        """
        if parameter_name not in self.space:
            raise ValueError(f"Parameter {parameter_name} not in architecture space.")
        
        return self.space[parameter_name]
    
    def sample_random_architecture(self):
        """
        Generate a random architecture configuration.
        
        Returns:
            dict: A complete architecture specification
        """
        # Select network type
        network_type = random.choice(self.space['network_type'])
        
        # Start with a random number of layers
        num_layers = random.choice(self.space['num_layers'])
        
        # Initialize the architecture with common parameters
        architecture = {
            'network_type': network_type,
            'num_layers': num_layers,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'learning_rate': random.choice(self.space['learning_rate']),
            'use_batch_norm': random.choice(self.space['use_batch_norm']),
            'dropout_rate': random.choice(self.space['dropout_rate']),
            'optimizer': random.choice(self.space['optimizer']),
        }
        
        # Add network-specific parameters based on the type
        if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
            # For CNN-based architectures
            architecture.update({
                'filters': [random.choice(self.space['filters']) for _ in range(num_layers)],
                'kernel_sizes': [random.choice(self.space['kernel_sizes']) for _ in range(num_layers)],
                'activations': [random.choice(self.space['activations']) for _ in range(num_layers)],
                'use_skip_connections': [random.choice(self.space['use_skip_connections']) for _ in range(num_layers)],
            })
            
            # Add MobileNet specific parameters
            if network_type == 'mobilenet':
                architecture['width_multiplier'] = random.choice(self.space['width_multiplier'])
            # Add DenseNet specific parameters
            elif network_type == 'densenet':
                architecture['growth_rate'] = random.choice([12, 24, 32, 48])
                architecture['block_config'] = random.choice([[6, 12, 24, 16], [6, 12, 32, 32], [6, 12, 48, 32]])
                architecture['compression_factor'] = random.choice([0.5, 0.7, 0.8])
                architecture['bn_size'] = random.choice([2, 4])
            # Add ShuffleNetV2 specific parameters
            elif network_type == 'shufflenetv2':
                architecture['width_multiplier'] = random.choice([0.5, 0.75, 1.0, 1.25, 1.5])
                # Define channel configurations for different model sizes
                if architecture['width_multiplier'] <= 0.5:
                    architecture['out_channels'] = [36, 72, 144, 576]
                elif architecture['width_multiplier'] <= 1.0:
                    architecture['out_channels'] = [116, 232, 464, 1024]
                else:
                    architecture['out_channels'] = [176, 352, 704, 1408]
                architecture['num_blocks_per_stage'] = random.choice([[2, 4, 2], [3, 7, 3], [4, 8, 4]])
            # Add EfficientNet specific parameters
            elif network_type == 'efficientnet':
                architecture['width_factor'] = random.choice([0.5, 0.75, 1.0, 1.25, 1.5])
                architecture['depth_factor'] = random.choice([0.8, 1.0, 1.2, 1.4])
                architecture['se_ratio'] = random.choice([0.0, 0.125, 0.25])
                
        elif network_type == 'mlp' or network_type == 'enhanced_mlp':
            # For MLP architecture
            architecture.update({
                'hidden_units': [random.choice(self.space['hidden_units']) for _ in range(num_layers)],
                'activations': [random.choice(self.space['activations']) for _ in range(num_layers)],
            })
            
            # Add Enhanced MLP specific parameters
            if network_type == 'enhanced_mlp':
                architecture['use_residual'] = random.choice([True, False])
                architecture['use_layer_norm'] = random.choice([True, False])
                # Add GELU to possible activations for enhanced MLP
                gelu_option = random.random() < 0.3  # 30% chance to use GELU
                if gelu_option:
                    idx = random.randint(0, num_layers-1)
                    architecture['activations'][idx] = 'gelu'
        
        return architecture
    
    def validate_architecture(self, architecture):
        """
        Check if an architecture is valid.
        
        Args:
            architecture: Architecture configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check if all required parameters are present
        required_params = ['network_type', 'num_layers', 'learning_rate', 'optimizer']
        
        for param in required_params:
            if param not in architecture:
                return False
        
        # Get network type
        network_type = architecture.get('network_type')
        if not isinstance(network_type, str):
            return False
            
        # Check network-specific parameters
        if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
            # For CNN-based architectures
            cnn_params = ['filters', 'kernel_sizes', 'activations']
            for param in cnn_params:
                if param not in architecture:
                    return False
                
            # Check if layer-specific parameters have the correct length
            num_layers = architecture['num_layers']
            for param in cnn_params:
                if param in architecture and len(architecture[param]) != num_layers:
                    return False
                    
            # Additional check for use_skip_connections
            if 'use_skip_connections' in architecture:
                if isinstance(architecture['use_skip_connections'], bool):
                    # Convert bool to list if needed before validating
                    architecture['use_skip_connections'] = [architecture['use_skip_connections']] * num_layers
                elif len(architecture['use_skip_connections']) != num_layers:
                    return False
            
            # Check specific architecture requirements
            if network_type == 'mobilenet' and 'width_multiplier' not in architecture:
                return False
            elif network_type == 'densenet':
                for param in ['growth_rate', 'block_config', 'compression_factor', 'bn_size']:
                    if param not in architecture:
                        return False
            elif network_type == 'shufflenetv2':
                for param in ['width_multiplier', 'out_channels', 'num_blocks_per_stage']:
                    if param not in architecture:
                        return False
            elif network_type == 'efficientnet':
                for param in ['width_factor', 'depth_factor', 'se_ratio']:
                    if param not in architecture:
                        return False
                    
        elif network_type == 'mlp' or network_type == 'enhanced_mlp':
            # For MLP architecture
            if 'hidden_units' not in architecture or 'activations' not in architecture:
                return False
            
            # Check if layer-specific parameters have the correct length
            num_layers = architecture['num_layers']
            for param in ['hidden_units', 'activations']:
                if param in architecture and len(architecture[param]) != num_layers:
                    return False
                    
            # Check Enhanced MLP specific parameters
            if network_type == 'enhanced_mlp':
                for param in ['use_residual', 'use_layer_norm']:
                    if param not in architecture:
                        return False
            num_layers = architecture['num_layers']
            if len(architecture['hidden_units']) != num_layers or len(architecture['activations']) != num_layers:
                return False
        
        return True
    
    def mutate_architecture(self, architecture, mutation_rate=0.2):
        """
        Apply random mutations to an architecture.
        
        Args:
            architecture: Architecture to mutate
            mutation_rate: Probability of mutation for each parameter
            
        Returns:
            dict: Mutated architecture
        """
        mutated = copy.deepcopy(architecture)
        network_type = mutated['network_type']
        
        # Try to mutate each parameter with probability mutation_rate
        for param in mutated:
            if random.random() < mutation_rate:
                if param == 'network_type':
                    # Change network type (rare mutation)
                    if random.random() < 0.2:  # Lower probability for this major change
                        mutated['network_type'] = random.choice(self.space['network_type'])
                        # If network type changed, we need to recreate the architecture
                        if mutated['network_type'] != network_type:
                            # Keep some global parameters
                            globals_to_keep = {
                                'input_shape': mutated['input_shape'],
                                'num_classes': mutated['num_classes'],
                                'learning_rate': mutated['learning_rate'],
                                'dropout_rate': mutated['dropout_rate'],
                                'use_batch_norm': mutated['use_batch_norm'],
                                'optimizer': mutated['optimizer'],
                                'num_layers': mutated['num_layers'],
                                'network_type': mutated['network_type']
                            }
                            # Generate new architecture of the new type
                            mutated = self.sample_random_architecture()
                            # Restore global parameters
                            for key, value in globals_to_keep.items():
                                mutated[key] = value
                            return mutated
                
                elif param == 'num_layers':
                    # Change number of layers within constraints
                    current_layers = mutated['num_layers']
                    mutated['num_layers'] = max(2, min(8, current_layers + random.choice([-1, 1])))
                    
                    # Adjust layer-specific parameters to match new layer count
                    if network_type == 'cnn' or network_type == 'resnet' or network_type == 'mobilenet':
                        for layer_param in ['filters', 'kernel_sizes', 'activations', 'use_skip_connections']:
                            if layer_param in mutated:
                                if mutated['num_layers'] > current_layers:
                                    # Add a layer
                                    for _ in range(mutated['num_layers'] - current_layers):
                                        mutated[layer_param].append(random.choice(self.space[layer_param]))
                                else:
                                    # Remove layers
                                    for _ in range(current_layers - mutated['num_layers']):
                                        mutated[layer_param].pop()
                                        
                    elif network_type == 'mlp':
                        for layer_param in ['hidden_units', 'activations']:
                            if layer_param in mutated:
                                if mutated['num_layers'] > current_layers:
                                    # Add a layer
                                    for _ in range(mutated['num_layers'] - current_layers):
                                        mutated[layer_param].append(random.choice(self.space[layer_param]))
                                else:
                                    # Remove layers
                                    for _ in range(current_layers - mutated['num_layers']):
                                        mutated[layer_param].pop()
                
                elif param in ['input_shape', 'num_classes']:
                    # These should not be mutated
                    continue
                
                elif isinstance(mutated[param], list) and param != 'input_shape':
                    # For list parameters, mutate a random element
                    if param in self.space:  # Only mutate if we have options
                        for i in range(len(mutated[param])):
                            if random.random() < mutation_rate:
                                mutated[param][i] = random.choice(self.space[param])
                
                else:
                    # For scalar parameters, replace with a random value
                    if param in self.space:  # Only mutate if we have options
                        mutated[param] = random.choice(self.space[param])
        
        # Fix use_skip_connections if it's a boolean instead of a list
        if 'use_skip_connections' in mutated and not isinstance(mutated['use_skip_connections'], list):
            mutated['use_skip_connections'] = [mutated['use_skip_connections']] * mutated['num_layers']
        
        return mutated