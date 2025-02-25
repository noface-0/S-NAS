"""
Evaluator for S-NAS

This module trains and evaluates candidate architectures on specified datasets.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluates neural network architectures on a dataset."""
    
    def __init__(self, dataset_registry, model_builder, device=None, 
                 max_epochs=10, patience=3, log_interval=10):
        """
        Initialize the evaluator.
        
        Args:
            dataset_registry: Registry containing datasets
            model_builder: Builder for creating models from architectures
            device: Device to use for training ('cuda' or 'cpu')
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience
            log_interval: How often to log training progress
        """
        self.dataset_registry = dataset_registry
        self.model_builder = model_builder
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_interval = log_interval
    
    def evaluate(self, architecture, dataset_name, fast_mode=False) -> Dict[str, Any]:
        """
        Train and evaluate a model architecture on a dataset.
        
        Args:
            architecture: Dictionary specifying the architecture
            dataset_name: Name of the dataset to use
            fast_mode: If True, use a reduced dataset and fewer epochs for faster evaluation
            
        Returns:
            dict: Evaluation results
        """
        # Get dataset configuration
        dataset_config = self.dataset_registry.get_dataset_config(dataset_name)
        
        # Complete architecture with dataset-specific values
        complete_architecture = architecture.copy()
        if 'input_shape' not in complete_architecture:
            complete_architecture['input_shape'] = dataset_config['input_shape']
        if 'num_classes' not in complete_architecture:
            complete_architecture['num_classes'] = dataset_config['num_classes']
        
        # Get data loaders
        train_loader, val_loader, test_loader = self.dataset_registry.get_dataset(dataset_name)
        
        # Build model, optimizer, and criterion
        model, optimizer, criterion = self.model_builder.build_training_components(complete_architecture)
        
        # Set up early stopping
        best_val_acc = 0.0
        patience_counter = 0
        
        # Adjust max epochs for fast mode
        max_epochs = 2 if fast_mode else self.max_epochs
        
        # Track metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Record start time
        start_time = time.time()
        
        # Training loop
        for epoch in range(max_epochs):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion, epoch, fast_mode
            )
            
            # Evaluate on validation set
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            # Save metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Log progress
            logger.info(f'Epoch {epoch+1}/{max_epochs} - '
                      f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, '
                      f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
            
            # Check early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save the best model weights (in memory)
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model for final evaluation
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        test_loss, test_acc = self._validate(model, test_loader, criterion)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Return evaluation results
        results = {
            'architecture': complete_architecture,
            'dataset': dataset_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'epochs_trained': len(train_losses),
            'training_time': training_time,
            'model_size': self._calculate_model_size(model),
            'flops_estimate': self._estimate_flops(model, dataset_config['input_shape'])
        }
        
        return results
    
    def _train_epoch(self, model, train_loader, optimizer, criterion, epoch, fast_mode):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Limit batches in fast mode
        batch_limit = 10 if fast_mode else len(train_loader)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= batch_limit:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % self.log_interval == 0:
                logger.debug(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        # Calculate epoch metrics
        avg_loss = total_loss / min(batch_limit, len(train_loader))
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _validate(self, model, data_loader, criterion):
        """Evaluate model on validation or test data."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Accumulate loss
                total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _calculate_model_size(self, model):
        """Calculate the size of a model in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def _estimate_flops(self, model, input_shape):
        """Estimate FLOPs for a given model (simplified)."""
        # This is a very simple estimate that just counts parameters as a proxy
        # For a more accurate estimate, one would use a proper profiler
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Simple heuristic: multiply params by input spatial dimensions
        # This is not accurate but gives a rough order of magnitude
        input_elements = input_shape[1] * input_shape[2]  # height * width
        flops_estimate = params * input_elements
        
        return flops_estimate