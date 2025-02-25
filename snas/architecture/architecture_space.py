"""
Architecture Space for S-NAS

This module defines the space of possible neural network architectures
that can be explored during the search process.
"""

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
            # Number of layers in the network
            'num_layers': [2, 3, 4, 5, 6, 7, 8],
            
            # Possible number of filters (neurons) for each layer
            'filters': [16, 32, 64, 128, 256],
            
            # Possible kernel sizes for conv layers
            'kernel_sizes': [3, 5, 7],
            
            # Possible activation functions
            'activations': ['relu', 'leaky_relu', 'elu', 'selu'],
            
            # Possible learning rates
            'learning_rate': [0.1, 0.01, 0.001, 0.0001],
            
            # Possible batch normalization options
            'use_batch_norm': [True, False],
            
            # Possible dropout options
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
            
            # Possible skip connection options
            'use_skip_connections': [True, False],
            
            # Possible optimizers
            'optimizer': ['sgd', 'adam', 'adamw'],
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
        # Start with a random number of layers
        num_layers = random.choice(self.space['num_layers'])
        
        # Initialize the architecture
        architecture = {
            'num_layers': num_layers,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            
            # Sample layer-specific parameters
            'filters': [random.choice(self.space['filters']) for _ in range(num_layers)],
            'kernel_sizes': [random.choice(self.space['kernel_sizes']) for _ in range(num_layers)],
            'activations': [random.choice(self.space['activations']) for _ in range(num_layers)],
            'use_skip_connections': [random.choice(self.space['use_skip_connections']) for _ in range(num_layers)],
            
            # Sample global parameters
            'learning_rate': random.choice(self.space['learning_rate']),
            'use_batch_norm': random.choice(self.space['use_batch_norm']),
            'dropout_rate': random.choice(self.space['dropout_rate']),
            'optimizer': random.choice(self.space['optimizer']),
        }
        
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
        required_params = ['num_layers', 'filters', 'kernel_sizes', 
                           'activations', 'learning_rate', 'optimizer']
        
        for param in required_params:
            if param not in architecture:
                return False
        
        # Check if layer-specific parameters have the correct length
        layer_params = ['filters', 'kernel_sizes', 'activations']
        num_layers = architecture['num_layers']
        
        for param in layer_params:
            if param in architecture and len(architecture[param]) != num_layers:
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
        
        # Try to mutate each parameter with probability mutation_rate
        for param in mutated:
            if random.random() < mutation_rate:
                if param == 'num_layers':
                    # Change number of layers (with care)
                    current_layers = mutated['num_layers']
                    new_layers = max(2, min(8, current_layers + random.choice([-1, 1])))
                    mutated['num_layers'] = new_layers
                    
                    # Adjust layer-specific parameters
                    layer_params = ['filters', 'kernel_sizes', 'activations', 'use_skip_connections']
                    for layer_param in layer_params:
                        if layer_param in mutated:
                            if new_layers > current_layers:
                                # Add a layer
                                for _ in range(new_layers - current_layers):
                                    mutated[layer_param].append(
                                        random.choice(self.space[layer_param])
                                    )
                            else:
                                # Remove layers
                                for _ in range(current_layers - new_layers):
                                    mutated[layer_param].pop()
                
                elif param in ['input_shape', 'num_classes']:
                    # These should not be mutated
                    continue
                
                elif isinstance(mutated[param], list) and param != 'input_shape':
                    # For list parameters like filters, mutate a random element
                    if param in self.space:  # Only mutate if we have options
                        for i in range(len(mutated[param])):
                            if random.random() < mutation_rate:
                                mutated[param][i] = random.choice(self.space[param])
                
                else:
                    # For scalar parameters, replace with a random value
                    if param in self.space:  # Only mutate if we have options
                        mutated[param] = random.choice(self.space[param])
        
        return mutated