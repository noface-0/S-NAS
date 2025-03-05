"""
RNN Controller for PNAS+ENAS hybrid.

This module implements an LSTM-based controller that learns to generate 
architecture decisions similar to the original ENAS paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

def safe_softmax(logits, dim=-1, temperature=1.0):
    """
    Safely compute softmax with checks for NaN and negative values.
    
    Args:
        logits: Input logits tensor
        dim: Dimension along which to apply softmax
        temperature: Temperature for softmax
        
    Returns:
        Probability distribution tensor
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Apply softmax
    probs = F.softmax(scaled_logits, dim=dim)
    
    # Check for NaN or negative probabilities - safely handle tensor truth value
    has_nan = torch.isnan(probs).any().item() if isinstance(torch.isnan(probs).any(), torch.Tensor) else torch.isnan(probs).any()
    has_negative = (probs < 0).any().item() if isinstance((probs < 0).any(), torch.Tensor) else (probs < 0).any()
    
    if has_nan or has_negative:
        logger.warning("Invalid probabilities detected, fixing before sampling")
        # Replace NaN or negative values with small positive values
        probs = torch.where(torch.isnan(probs) | (probs < 0), 
                            torch.tensor(1e-8, device=logits.device), 
                            probs)
        # Re-normalize
        probs = probs / probs.sum(dim=dim, keepdim=True)
    
    return probs

class RNNController(nn.Module):
    """
    LSTM-based controller for architecture generation.
    
    The controller generates architecture decisions sequentially, similar to
    the ENAS controller but modified to work within the PNAS framework.
    """
    
    def __init__(self, 
                 input_size: int = 8,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 output_size: int = None,  # Will be set based on action spaces
                 temperature: float = 1.0,
                 tanh_constant: float = 1.5,
                 entropy_weight: float = 0.0001,
                 device: str = None):
        """
        Initialize the RNN controller.
        
        Args:
            input_size: Size of the input embedding
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            output_size: Size of output logits (usually number of actions)
            temperature: Temperature for sampling actions
            tanh_constant: Constant for scaling logits
            entropy_weight: Weight for entropy regularization
            device: Device to run on ('cuda' or 'cpu')
        """
        super(RNNController, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.entropy_weight = entropy_weight
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create action spaces for different architecture decisions
        self.action_spaces = {
            'num_layers': list(range(3, 21)),                            # 3-20 layers
            'network_type': ['cnn', 'mlp', 'enhanced_mlp', 'resnet',
                            'mobilenet', 'densenet', 'shufflenetv2'],
            'filters': [16, 32, 64, 128, 256, 512],
            'kernel_sizes': [1, 3, 5, 7],
            'activations': ['relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
            'use_batch_norm': [True, False],
            'use_skip_connections': [True, False],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'hidden_units': [64, 128, 256, 512, 1024, 2048]
        }
        
        # Core LSTM controller
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Create embeddings for categorical inputs
        self.embeddings = nn.ModuleDict({
            'network_type': nn.Embedding(len(self.action_spaces['network_type']), input_size),
            'layer_type': nn.Embedding(5, input_size)  # Different layer types (conv, pool, skip, etc)
        })
        
        # Create output heads for different decisions
        self.decision_heads = nn.ModuleDict({
            'network_type': nn.Linear(hidden_size, len(self.action_spaces['network_type'])),
            'num_layers': nn.Linear(hidden_size, len(self.action_spaces['num_layers'])),
            'filters': nn.Linear(hidden_size, len(self.action_spaces['filters'])),
            'kernel_sizes': nn.Linear(hidden_size, len(self.action_spaces['kernel_sizes'])),
            'activations': nn.Linear(hidden_size, len(self.action_spaces['activations'])),
            'use_batch_norm': nn.Linear(hidden_size, 2),  # Binary choice
            'use_skip_connections': nn.Linear(hidden_size, 2),  # Binary choice
            'dropout_rate': nn.Linear(hidden_size, len(self.action_spaces['dropout_rate'])),
            'hidden_units': nn.Linear(hidden_size, len(self.action_spaces['hidden_units']))
        })
        
        # Move to device
        self.to(self.device)
        
        # Training history
        self.rewards = []
        self.log_probs = []
        self.entropies = []
    
    def reset_history(self):
        """Reset controller history for a new training episode."""
        self.rewards = []
        self.log_probs = []
        self.entropies = []
    
    def forward(self, inputs=None, hidden=None):
        """
        Forward pass through the controller.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, input_size] or None
            hidden: Initial hidden state or None
            
        Returns:
            outputs: LSTM outputs
            hidden: Final hidden state
        """
        if inputs is None:
            # Use a dummy input if none provided (batch_size=1)
            inputs = torch.zeros(1, 1, self.input_size).to(self.device)
            
        outputs, hidden = self.lstm(inputs, hidden)
        return outputs, hidden
    
    def sample_architecture(self, complexity_level=1) -> Tuple[Dict[str, Any], List[float], List[float]]:
        """
        Sample an architecture from the controller.
        
        Args:
            complexity_level: Current complexity level for PNAS
            
        Returns:
            architecture: Dictionary specifying the sampled architecture
            log_probs: Log probabilities of the sampled actions
            entropies: Entropies of the action distributions
        """
        # Initialize empty architecture
        architecture = {}
        log_probs = []
        entropies = []
        
        # Initial inputs (all zeros)
        inputs = torch.zeros(1, 1, self.input_size).to(self.device)
        hidden = None
        
        # Sample network type first
        outputs, hidden = self.forward(inputs, hidden)
        logits = self.decision_heads['network_type'](outputs[:, -1])
        scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
        
        # Sample from the distribution with safety checks
        probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
        action_idx = torch.multinomial(probs, 1).item()
        
        # Record the log prob and entropy
        log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # Set the network type
        network_type = self.action_spaces['network_type'][action_idx]
        architecture['network_type'] = network_type
        
        # Get embedding for the network type
        network_type_idx = self.action_spaces['network_type'].index(network_type)
        network_type_embedding = self.embeddings['network_type'](torch.tensor([network_type_idx]).to(self.device))
        
        # Use this as input for the next decision
        inputs = network_type_embedding.unsqueeze(1)
        
        # Sample number of layers based on complexity level
        outputs, hidden = self.forward(inputs, hidden)
        logits = self.decision_heads['num_layers'](outputs[:, -1])
        scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
        
        # Adjust for complexity level
        if complexity_level == 1:
            # Limit to fewer layers for low complexity
            max_idx = min(4, len(self.action_spaces['num_layers']))
            mask = torch.zeros_like(scaled_logits)
            mask[0, :max_idx] = 1.0
            scaled_logits = scaled_logits * mask - 1e9 * (1 - mask)
        elif complexity_level == 2:
            # Medium complexity - allow more layers
            max_idx = min(8, len(self.action_spaces['num_layers']))
            mask = torch.zeros_like(scaled_logits)
            mask[0, :max_idx] = 1.0
            scaled_logits = scaled_logits * mask - 1e9 * (1 - mask)
        
        # Sample from the distribution with safety checks
        probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0) 
        action_idx = torch.multinomial(probs, 1).item()
        
        # Record the log prob and entropy safely
        log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # Set the number of layers
        num_layers = self.action_spaces['num_layers'][action_idx]
        architecture['num_layers'] = num_layers
        
        # Sample global parameters
        # Dropout rate
        outputs, hidden = self.forward(inputs, hidden)
        logits = self.decision_heads['dropout_rate'](outputs[:, -1])
        scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
        
        probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
        action_idx = torch.multinomial(probs, 1).item()
        
        log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # Set the dropout rate
        architecture['dropout_rate'] = self.action_spaces['dropout_rate'][action_idx]
        
        # Batch normalization
        outputs, hidden = self.forward(inputs, hidden)
        logits = self.decision_heads['use_batch_norm'](outputs[:, -1])
        scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
        
        probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
        action_idx = torch.multinomial(probs, 1).item()
        
        log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # Set batch normalization
        architecture['use_batch_norm'] = bool(action_idx)
        
        # Now sample parameters for each layer
        if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
            # CNN-specific parameters
            filters = []
            kernel_sizes = []
            activations = []
            skip_connections = []
            
            # Sample parameters for each layer
            for layer_idx in range(num_layers):
                # Layer embedding (position encoding)
                layer_pos = torch.tensor([min(layer_idx / 20.0, 1.0)]).to(self.device).unsqueeze(0).unsqueeze(0)
                inputs = torch.cat([inputs, layer_pos], dim=-1)[:, :, :self.input_size]
                
                # Sample filters
                outputs, hidden = self.forward(inputs, hidden)
                logits = self.decision_heads['filters'](outputs[:, -1])
                scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
                
                probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
                action_idx = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                filters.append(self.action_spaces['filters'][action_idx])
                
                # Sample kernel sizes
                outputs, hidden = self.forward(inputs, hidden)
                logits = self.decision_heads['kernel_sizes'](outputs[:, -1])
                scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
                
                probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
                action_idx = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                kernel_sizes.append(self.action_spaces['kernel_sizes'][action_idx])
                
                # Sample activations
                outputs, hidden = self.forward(inputs, hidden)
                logits = self.decision_heads['activations'](outputs[:, -1])
                scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
                
                probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
                action_idx = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                activations.append(self.action_spaces['activations'][action_idx])
                
                # Sample skip connections (only for layers after the first)
                if layer_idx > 0 and (network_type in ['resnet'] or complexity_level > 1):
                    outputs, hidden = self.forward(inputs, hidden)
                    logits = self.decision_heads['use_skip_connections'](outputs[:, -1])
                    scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
                    
                    probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
                    action_idx = torch.multinomial(probs, 1).item()
                    
                    log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
                    entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                    
                    skip_connections.append(bool(action_idx))
                else:
                    skip_connections.append(False)
            
            # Add layer parameters to architecture
            architecture['filters'] = filters
            architecture['kernel_sizes'] = kernel_sizes
            architecture['activations'] = activations
            architecture['use_skip_connections'] = skip_connections
            
        elif network_type in ['mlp', 'enhanced_mlp']:
            # MLP-specific parameters
            hidden_units = []
            activations = []
            
            # Sample parameters for each layer
            for layer_idx in range(num_layers):
                # Layer embedding (position encoding)
                layer_pos = torch.tensor([min(layer_idx / 20.0, 1.0)]).to(self.device).unsqueeze(0).unsqueeze(0)
                inputs = torch.cat([inputs, layer_pos], dim=-1)[:, :, :self.input_size]
                
                # Sample hidden units
                outputs, hidden = self.forward(inputs, hidden)
                logits = self.decision_heads['hidden_units'](outputs[:, -1])
                scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
                
                probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
                action_idx = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                hidden_units.append(self.action_spaces['hidden_units'][action_idx])
                
                # Sample activations
                outputs, hidden = self.forward(inputs, hidden)
                logits = self.decision_heads['activations'](outputs[:, -1])
                scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
                
                probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
                action_idx = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                activations.append(self.action_spaces['activations'][action_idx])
            
            # Add layer parameters to architecture
            architecture['hidden_units'] = hidden_units
            architecture['activations'] = activations
            
            # Add special parameters for enhanced MLP
            if network_type == 'enhanced_mlp':
                # Sample use_residual
                outputs, hidden = self.forward(inputs, hidden)
                logits = self.decision_heads['use_skip_connections'](outputs[:, -1])  # Reuse skip connection head
                scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
                
                probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
                action_idx = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                architecture['use_residual'] = bool(action_idx)
                
                # Sample use_layer_norm
                outputs, hidden = self.forward(inputs, hidden)
                logits = self.decision_heads['use_batch_norm'](outputs[:, -1])  # Reuse batch norm head
                scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
                
                probs = safe_softmax(scaled_logits, dim=-1, temperature=1.0)
                action_idx = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log(probs[:, action_idx] + 1e-8)  # Safe log
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                architecture['use_layer_norm'] = bool(action_idx)
        
        # Add input shape and num_classes placeholders to be filled by the search algorithm
        architecture['input_shape'] = (3, 32, 32)  # Default, will be updated
        architecture['num_classes'] = 10  # Default, will be updated
        
        # Always include required parameters to avoid validation failures
        if 'learning_rate' not in architecture:
            architecture['learning_rate'] = random.choice([0.1, 0.01, 0.001, 0.0001])
        if 'optimizer' not in architecture:
            architecture['optimizer'] = random.choice(['sgd', 'adam', 'adamw', 'rmsprop'])
        
        return architecture, log_probs, entropies
    
    def update_policy(self, rewards, learning_rate=0.00035):
        """
        Update the controller policy using REINFORCE.
        
        Args:
            rewards: List of rewards for each sampled architecture
            learning_rate: Learning rate for policy update
        """
        if not self.log_probs or not rewards:
            logger.warning("No data available for policy update")
            return
        
        # Convert rewards to tensor
        rewards = torch.tensor(rewards).to(self.device)
        
        # Normalize rewards
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = 0
        for log_prob, reward in zip(self.log_probs, rewards):
            policy_loss -= log_prob * reward
        
        # Add entropy regularization
        entropy_loss = 0
        for entropy in self.entropies:
            entropy_loss -= entropy
        entropy_loss *= self.entropy_weight
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        
        optimizer.step()
        
        # Reset history
        self.reset_history()
        
        return loss.item()
    
    def save_model(self, path: str) -> None:
        """
        Save the controller model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'action_spaces': self.action_spaces,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'temperature': self.temperature,
            'tanh_constant': self.tanh_constant,
            'entropy_weight': self.entropy_weight
        }, path)
        
        logger.info(f"Controller model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load the controller model from a file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update model parameters
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.temperature = checkpoint['temperature']
        self.tanh_constant = checkpoint['tanh_constant']
        self.entropy_weight = checkpoint['entropy_weight']
        self.action_spaces = checkpoint['action_spaces']
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Controller model loaded from {path}")