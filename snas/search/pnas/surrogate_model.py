"""
Surrogate Model for PNAS

This module implements the surrogate model used in PNAS to predict the performance
of neural network architectures without fully training them. The original paper
uses an LSTM-based predictor that takes architecture encodings as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Union

logger = logging.getLogger(__name__)

class SurrogateModel(nn.Module):
    """
    LSTM-based surrogate model to predict architecture performance.
    
    This model takes architecture encodings and predicts validation accuracy
    without the need for full training. It's trained on architecture-performance
    pairs collected during the search process.
    """
    
    def __init__(self, 
                 input_size: int = 32, 
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 device: str = None):
        """
        Initialize the surrogate model.
        
        Args:
            input_size: Size of the architecture encoding vectors
            hidden_size: Size of the LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            device: Device to use for computation
        """
        super(SurrogateModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)  # Predicts a single performance value
        
        # Move to device
        self.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.is_trained = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Architecture encodings [batch_size, seq_len, input_size]
            
        Returns:
            Predicted performance values [batch_size, 1]
        """
        # Process through LSTM
        out, _ = self.lstm(x)
        
        # Use the last output from the sequence
        out = out[:, -1, :]
        
        # Process through prediction layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Output is a single performance prediction value per architecture
        return out
    
    def encode_architecture(self, architecture: Dict[str, Any]) -> torch.Tensor:
        """
        Encode an architecture as a tensor for input to the surrogate model.
        
        Args:
            architecture: Dictionary specifying the architecture
            
        Returns:
            Tensor encoding of the architecture [1, seq_len, input_size]
        """
        try:
            # Extract relevant architecture parameters
            network_type = architecture.get('network_type', 'cnn')
            num_layers = architecture.get('num_layers', 0)
            
            # Initialize encoding
            encoding = []
            
            # Add network type as one-hot encoding (8 possible types as listed in architecture_space.py)
            network_types = ['cnn', 'mlp', 'enhanced_mlp', 'resnet', 'mobilenet', 
                             'densenet', 'shufflenetv2', 'efficientnet']
            network_type_onehot = [1 if t == network_type else 0 for t in network_types]
            encoding.extend(network_type_onehot)
            
            # Add normalized number of layers
            encoding.append(num_layers / 50.0)  # Normalize by max possible layers
            
            # Add global parameters
            encoding.append(float(architecture.get('use_batch_norm', False)))
            encoding.append(architecture.get('dropout_rate', 0.0))
            
            # Add learning rate (normalized)
            lr_map = {0.1: 0.8, 0.01: 0.6, 0.001: 0.4, 0.0001: 0.2, 0.00001: 0.0}
            encoding.append(lr_map.get(architecture.get('learning_rate', 0.001), 0.4))
            
            # Add optimizer as one-hot
            optimizers = ['sgd', 'adam', 'adamw', 'rmsprop']
            opt_onehot = [1 if o == architecture.get('optimizer', 'adam') else 0 
                          for o in optimizers]
            encoding.extend(opt_onehot)
            
            # Add layer-specific parameters
            if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
                # For CNN-based architectures
                filters = architecture.get('filters', [])
                kernel_sizes = architecture.get('kernel_sizes', [])
                activations = architecture.get('activations', [])
                skip_connections = architecture.get('use_skip_connections', [])
                
                # Handle non-list skip connections
                if not isinstance(skip_connections, list):
                    skip_connections = [skip_connections] * min(num_layers, 10)
                
                # Process up to 10 layers (for fixed-size encoding)
                layer_encodings = []
                for i in range(min(num_layers, 10)):
                    layer_enc = []
                    
                    # Add normalized filter count
                    if i < len(filters):
                        layer_enc.append(filters[i] / 1024.0)  # Normalize by max filters
                    else:
                        layer_enc.append(0.0)
                    
                    # Add normalized kernel size
                    if i < len(kernel_sizes):
                        layer_enc.append(kernel_sizes[i] / 9.0)  # Normalize by max kernel size
                    else:
                        layer_enc.append(0.0)
                    
                    # Add activation as one-hot
                    activation_types = ['relu', 'leaky_relu', 'elu', 'selu', 'gelu']
                    act_onehot = [0] * len(activation_types)
                    if i < len(activations):
                        try:
                            act_idx = activation_types.index(activations[i]) if activations[i] in activation_types else 0
                            act_onehot[act_idx] = 1
                        except (ValueError, IndexError):
                            # Default to first activation if there's an issue
                            act_onehot[0] = 1
                    layer_enc.extend(act_onehot)
                    
                    # Add skip connection
                    if i < len(skip_connections):
                        try:
                            layer_enc.append(float(skip_connections[i]))
                        except (TypeError, ValueError):
                            layer_enc.append(0.0)
                    else:
                        layer_enc.append(0.0)
                    
                    layer_encodings.append(layer_enc)
                
                # Pad to 10 layers if needed
                while len(layer_encodings) < 10:
                    layer_encodings.append([0.0] * (2 + len(activation_types) + 1))  # Same size as layer encoding
                
                # Flatten layer encodings
                for layer_enc in layer_encodings:
                    encoding.extend(layer_enc)
                    
            elif network_type in ['mlp', 'enhanced_mlp']:
                # For MLP architectures
                hidden_units = architecture.get('hidden_units', [])
                activations = architecture.get('activations', [])
                
                # Add enhanced MLP specific parameters
                if network_type == 'enhanced_mlp':
                    encoding.append(float(architecture.get('use_residual', False)))
                    encoding.append(float(architecture.get('use_layer_norm', False)))
                else:
                    encoding.extend([0.0, 0.0])  # Padding for regular MLPs
                
                # Process up to 10 layers
                layer_encodings = []
                for i in range(min(num_layers, 10)):
                    layer_enc = []
                    
                    # Add normalized hidden units
                    if i < len(hidden_units):
                        layer_enc.append(hidden_units[i] / 4096.0)  # Normalize by max units
                    else:
                        layer_enc.append(0.0)
                    
                    # Add activation as one-hot
                    activation_types = ['relu', 'leaky_relu', 'elu', 'selu', 'gelu']
                    act_onehot = [0] * len(activation_types)
                    if i < len(activations):
                        try:
                            act_idx = activation_types.index(activations[i]) if activations[i] in activation_types else 0
                            act_onehot[act_idx] = 1
                        except (ValueError, IndexError):
                            # Default to first activation if there's an issue
                            act_onehot[0] = 1
                    layer_enc.extend(act_onehot)
                    
                    layer_encodings.append(layer_enc)
                
                # Pad to 10 layers if needed
                while len(layer_encodings) < 10:
                    layer_encodings.append([0.0] * (1 + len(activation_types)))  # Same size as layer encoding
                
                # Flatten layer encodings
                for layer_enc in layer_encodings:
                    encoding.extend(layer_enc)
            
            # Convert to tensor and add batch/sequence dimensions
            # Explicitly move tensor to the device
            tensor_encoding = torch.tensor([encoding], dtype=torch.float32).to(self.device)
            
            # Reshape to [batch_size, seq_len, input_size]
            # We'll treat each segment of the encoding as a "timestep" in the sequence
            seq_len = 12  # Network type + global params + 10 layers
            
            # Calculate the input_size based on the encoding vector
            actual_input_size = len(encoding) // seq_len
            
            # If encoding doesn't divide evenly, pad it
            if len(encoding) % seq_len != 0:
                padding_needed = seq_len - (len(encoding) % seq_len)
                padding_tensor = torch.zeros((1, padding_needed), device=self.device)
                tensor_encoding = torch.cat([tensor_encoding, padding_tensor], dim=1)
                actual_input_size = len(tensor_encoding[0]) // seq_len
            
            # Ensure consistent feature size after the first model is trained
            if hasattr(self, 'is_trained') and self.is_trained and self.input_size != actual_input_size:
                logger.warning(f"Architecture encoding size mismatch: expected {self.input_size}, got {actual_input_size}")
                
                # Reshape the tensor to [batch_size, seq_len, actual_input_size]
                reshaped = tensor_encoding.view(1, seq_len, actual_input_size)
                
                # Pad or truncate to match expected input size
                if actual_input_size < self.input_size:
                    # Pad with zeros to match expected size
                    padding = torch.zeros((1, seq_len, self.input_size - actual_input_size), device=self.device)
                    return torch.cat([reshaped, padding], dim=2)
                else:
                    # Truncate to expected size
                    return reshaped[:, :, :self.input_size]
            
            # Reshape to [batch_size, seq_len, input_size]
            result = tensor_encoding.view(1, seq_len, actual_input_size)
            
            # Ensure the tensor is on the correct device
            if result.device != torch.device(self.device):
                logger.warning(f"Tensor on wrong device: {result.device}, moving to {self.device}")
                result = result.to(self.device)
                
            return result
            
        except Exception as e:
            logger.error(f"Error encoding architecture: {str(e)}")
            # Return a default encoding to avoid crashing
            # This creates a zero tensor with the expected shape
            if hasattr(self, 'is_trained') and self.is_trained:
                # If already trained, use the expected input size
                return torch.zeros((1, 12, self.input_size), device=self.device)
            else:
                # Otherwise use a reasonable default size
                return torch.zeros((1, 12, 32), device=self.device)
    
    def train_surrogate(self, 
                        architectures: List[Dict[str, Any]], 
                        performances: List[float],
                        num_epochs: int = 100,
                        batch_size: int = 32,
                        learning_rate: float = 0.001,
                        validation_split: float = 0.2,
                        progress_callback = None) -> None:
        """
        Train the surrogate model on architecture-performance pairs.
        
        Args:
            architectures: List of architecture dictionaries
            performances: List of performance values (e.g., validation accuracies)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation
            progress_callback: Optional callback function to report training progress.
                              Will be called with (current_epoch, total_epochs, current_loss)
        """
        if len(architectures) != len(performances):
            raise ValueError("Number of architectures and performances must match")
            
        if len(architectures) < 10:
            logger.warning("Training surrogate model with very few samples (<10)")
            
        # Reset training history
        self.train_losses = []
        self.val_losses = []
        
        # Log device information
        logger.info(f"Training surrogate model on device: {self.device}")
        
        try:
            # Encode all architectures
            encoded_architectures = []
            for arch in architectures:
                encoded_arch = self.encode_architecture(arch)
                encoded_architectures.append(encoded_arch)
                
            # Verify encoded architectures dimensions
            if encoded_architectures:
                sample_shape = encoded_architectures[0].shape
                logger.info(f"Encoded architecture shape: {sample_shape}")
                
                # Ensure all encodings have the same shape
                for i, enc in enumerate(encoded_architectures):
                    if enc.shape != sample_shape:
                        logger.warning(f"Architecture {i} has different shape: {enc.shape} vs {sample_shape}")
                        # Reshape/pad to match the expected dimensions
                        if enc.shape[1:] != sample_shape[1:]:
                            # Pad the sequence dimension (dim=1) and feature dimension (dim=2) if needed
                            padded_enc = torch.zeros(sample_shape, device=self.device)
                            min_seq_len = min(enc.shape[1], sample_shape[1])
                            min_feat_len = min(enc.shape[2], sample_shape[2])
                            padded_enc[:, :min_seq_len, :min_feat_len] = enc[:, :min_seq_len, :min_feat_len]
                            encoded_architectures[i] = padded_enc
            
            # Convert performances to tensor, explicitly moving to the specified device
            performance_tensor = torch.tensor(performances, dtype=torch.float32).to(self.device).view(-1, 1)
            
            # Create dataset
            num_samples = len(architectures)
            indices = np.random.permutation(num_samples)
            split_idx = int(np.floor(validation_split * num_samples))
            
            train_indices = indices[split_idx:]
            val_indices = indices[:split_idx]
            
            # Create optimizer
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            logger.info(f"Training surrogate model on {len(train_indices)} samples for {num_epochs} epochs")
            
            for epoch in range(num_epochs):
                self.train()
                running_loss = 0.0
                
                # Process in batches
                for i in range(0, len(train_indices), batch_size):
                    batch_indices = train_indices[i:i+batch_size]
                    
                    try:
                        # Create batch
                        batch_x = torch.cat([encoded_architectures[idx] for idx in batch_indices], dim=0)
                        batch_y = torch.cat([performance_tensor[idx] for idx in batch_indices], dim=0)
                        
                        # Ensure both tensors are on the same device
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        # Forward pass
                        outputs = self(batch_x)
                        
                        # Ensure output and target have same shape
                        if outputs.shape != batch_y.shape:
                            logger.warning(f"Shape mismatch: outputs={outputs.shape}, batch_y={batch_y.shape}")
                            # Reshape if needed
                            if outputs.dim() == 2 and batch_y.dim() == 2 and outputs.size(0) == batch_y.size(0):
                                # Only reshape if it's just a matter of different feature dimensions
                                outputs = outputs.view(batch_y.shape)
                        
                        loss = criterion(outputs, batch_y)
                        
                        # Backward pass and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item() * len(batch_indices)
                    except Exception as e:
                        logger.error(f"Error in batch {i}: {str(e)}")
                        continue
                
                # Calculate epoch loss
                epoch_loss = running_loss / len(train_indices) if train_indices else 0.0
                self.train_losses.append(epoch_loss)
                
                # Validation
                val_loss = 0.0
                if len(val_indices) > 0:
                    self.eval()
                    try:
                        with torch.no_grad():
                            val_x = torch.cat([encoded_architectures[idx] for idx in val_indices], dim=0)
                            val_y = torch.cat([performance_tensor[idx] for idx in val_indices], dim=0)
                            
                            # Ensure both tensors are on the same device
                            val_x = val_x.to(self.device)
                            val_y = val_y.to(self.device)
                            
                            val_outputs = self(val_x)
                            
                            # Reshape if needed
                            if val_outputs.shape != val_y.shape:
                                logger.warning(f"Validation shape mismatch: outputs={val_outputs.shape}, val_y={val_y.shape}")
                                if val_outputs.dim() == 2 and val_y.dim() == 2 and val_outputs.size(0) == val_y.size(0):
                                    val_outputs = val_outputs.view(val_y.shape)
                                    
                            val_loss = criterion(val_outputs, val_y).item()
                            self.val_losses.append(val_loss)
                            
                            # Log validation results
                            if (epoch + 1) % 10 == 0:
                                logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
                    except Exception as e:
                        logger.error(f"Error in validation: {str(e)}")
                        self.val_losses.append(float('inf'))
                else:
                    # Log progress without validation
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
                
                # Report progress if callback provided
                if progress_callback is not None:
                    progress_callback(epoch + 1, num_epochs, epoch_loss)
            
            # Mark as trained
            self.is_trained = True
            logger.info("Surrogate model training completed")
            
        except Exception as e:
            logger.error(f"Error during surrogate model training: {str(e)}")
            # Make sure the model is still usable even if training failed
            self.is_trained = False
    
    def predict_performance(self, architecture: Dict[str, Any]) -> float:
        """
        Predict the performance of an architecture.
        
        Args:
            architecture: Dictionary specifying the architecture
            
        Returns:
            Predicted performance value
        """
        if not self.is_trained:
            logger.warning("Surrogate model used for prediction before training")
            
        # Set to evaluation mode
        self.eval()
        
        # Encode the architecture
        encoded_arch = self.encode_architecture(architecture)
        
        # Make prediction
        with torch.no_grad():
            prediction = self(encoded_arch)
            
        # Return as float
        return prediction.item()
    
    def save_model(self, path: str) -> None:
        """
        Save the surrogate model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'is_trained': self.is_trained
        }, path)
        
        logger.info(f"Surrogate model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load the surrogate model from a file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update model parameters
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.is_trained = checkpoint['is_trained']
        
        logger.info(f"Surrogate model loaded from {path}")