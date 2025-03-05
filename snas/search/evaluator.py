import time
import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
import traceback
import os
import copy
from collections import defaultdict
from ..utils.exceptions import EvaluationError, ArchitectureError

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluates neural network architectures on a dataset."""
    
    def __init__(self, dataset_registry, model_builder, device=None, 
                 max_epochs=10, patience=3, log_interval=10, 
                 min_delta=0.001, monitor='val_acc', enable_weight_sharing=False,
                 weight_sharing_max_models=100):
        """
        Initialize the evaluator.
        
        Args:
            dataset_registry: Registry containing datasets
            model_builder: Builder for creating models from architectures
            device: Device to use for training ('cuda' or 'cpu')
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience (number of epochs without improvement)
            log_interval: How often to log training progress
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor for early stopping ('val_acc' or 'val_loss')
            enable_weight_sharing: Whether to enable weight sharing between models (ENAS approach)
            weight_sharing_max_models: Maximum number of models to keep in the weight sharing pool
        """
        self.dataset_registry = dataset_registry
        self.model_builder = model_builder
        
        # Explicitly check and set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Log device information
        logger.info(f"Evaluator initialized with device: {self.device}")
        if self.device.startswith('cuda'):
            # Log GPU information
            try:
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                    logger.info(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
                else:
                    logger.warning("CUDA device specified but CUDA is not available")
            except Exception as e:
                logger.warning(f"Could not get GPU information: {str(e)}")
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_interval = log_interval
        self.min_delta = min_delta
        self.monitor = monitor
        self.enable_weight_sharing = enable_weight_sharing
        self.weight_sharing_max_models = weight_sharing_max_models
        
        # Initialize shared parameter pools (organized by network type)
        # Each pool will contain model state dicts that can be used to initialize new models
        self.shared_param_pools = defaultdict(list)
        
        # Track model performance with different shared weights
        self.shared_weight_performance = defaultdict(list)
        
        # Create a container for weight sharing pool for PNAS+ENAS integration
        self.weight_sharing_pool = [] if enable_weight_sharing else None
        
    def estimate_performance_from_shared_weights(self, architecture, dataset_name, metric):
        """
        Estimate the performance of an architecture using the shared weights pool.
        This method is used by the PNAS+ENAS combined search to get quick estimates.
        
        Args:
            architecture: Architecture to estimate
            dataset_name: Name of the dataset
            metric: Metric to use for estimation ('val_acc', 'val_loss', etc.)
            
        Returns:
            float: Estimated performance value or None if no estimate is available
        """
        if not self.enable_weight_sharing or not self.weight_sharing_pool:
            logger.info("Shared weights not available for estimation")
            return None
            
        try:
            # Get dataset configuration
            dataset_config = self.dataset_registry.get_dataset_config(dataset_name)
            
            # Complete architecture with dataset-specific values
            complete_architecture = architecture.copy()
            if 'input_shape' not in complete_architecture:
                complete_architecture['input_shape'] = dataset_config['input_shape']
            if 'num_classes' not in complete_architecture:
                complete_architecture['num_classes'] = dataset_config['num_classes']
            
            # Build model
            model = self.model_builder.build_model(complete_architecture)
            model = model.to(self.device)
            
            # Get validation loader for evaluation
            _, val_loader, _ = self.dataset_registry.get_dataset(dataset_name)
            
            # Evaluate with each set of shared weights and average the results
            estimates = []
            
            # Use at most 5 sets of shared weights for efficiency
            for weights_info in self.weight_sharing_pool[:5]:
                if weights_info['network_type'] == complete_architecture.get('network_type', None):
                    # Load shared weights that match the network type
                    try:
                        # Try to load weights - this may fail if architectures are incompatible
                        model.load_state_dict(weights_info['state_dict'], strict=False)
                        
                        # Evaluate the model
                        model.eval()
                        val_loss = 0
                        correct = 0
                        total = 0
                        
                        with torch.no_grad():
                            for inputs, targets in val_loader:
                                inputs, targets = inputs.to(self.device), targets.to(self.device)
                                outputs = model(inputs)
                                
                                if isinstance(outputs, tuple):
                                    outputs = outputs[0]  # Handle models that return multiple outputs
                                
                                loss = torch.nn.functional.cross_entropy(outputs, targets)
                                val_loss += loss.item()
                                
                                _, predicted = outputs.max(1)
                                total += targets.size(0)
                                correct += predicted.eq(targets).sum().item()
                        
                        # Calculate metrics
                        avg_loss = val_loss / len(val_loader)
                        accuracy = correct / total
                        
                        # Use the appropriate metric
                        if metric == 'val_acc':
                            estimates.append(accuracy)
                        elif metric == 'val_loss':
                            estimates.append(avg_loss)
                        else:
                            # Default to accuracy for other metrics
                            estimates.append(accuracy)
                    except Exception as e:
                        # Skip incompatible weights
                        logger.debug(f"Couldn't use shared weights for estimation: {str(e)}")
                        continue
            
            # Return the average estimate if we have any
            if estimates:
                return sum(estimates) / len(estimates)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in shared weights estimation: {str(e)}")
            return None
    
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
        try:
            # Get dataset configuration
            dataset_config = self.dataset_registry.get_dataset_config(dataset_name)
            
            # Complete architecture with dataset-specific values
            complete_architecture = architecture.copy()
            if 'input_shape' not in complete_architecture:
                complete_architecture['input_shape'] = dataset_config['input_shape']
            if 'num_classes' not in complete_architecture:
                complete_architecture['num_classes'] = dataset_config['num_classes']
                
            # Fix any parameter types that need to be lists
            if 'use_skip_connections' in complete_architecture and not isinstance(complete_architecture['use_skip_connections'], list):
                complete_architecture['use_skip_connections'] = [complete_architecture['use_skip_connections']] * complete_architecture['num_layers']
            
            # Get data loaders
            train_loader, val_loader, test_loader = self.dataset_registry.get_dataset(dataset_name)
        except Exception as e:
            error_msg = f"Failed to prepare dataset for evaluation: {str(e)}"
            logger.error(error_msg)
            trace = traceback.format_exc()
            raise EvaluationError(error_msg, architecture=architecture, details=trace)
        
        # Build model, optimizer, and criterion (with optional weight sharing)
        try:
            if self.enable_weight_sharing:
                # Attempt to find compatible shared weights
                model, optimizer, criterion, used_shared_weights = self._build_with_weight_sharing(complete_architecture)
            else:
                # Build model without weight sharing
                model, optimizer, criterion = self.model_builder.build_training_components(complete_architecture)
                used_shared_weights = False
        except Exception as e:
            error_msg = f"Failed to build model: {str(e)}"
            logger.error(error_msg)
            trace = traceback.format_exc()
            raise ArchitectureError(error_msg, architecture=complete_architecture)
        
        # Set up early stopping
        best_val_acc = 0.0
        best_val_loss = float('inf')
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
        
        # Record start GPU time if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_start_time = torch.cuda.Event(enable_timing=True)
            gpu_end_time = torch.cuda.Event(enable_timing=True)
            gpu_start_time.record()
        
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
            
            # Check early stopping based on monitored metric
            improved = False
            
            # Always track both best accuracy and best loss, regardless of which one is used for early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Check improvement based on monitored metric
            if self.monitor == 'val_acc':
                if val_acc - best_val_acc > -self.min_delta:  # Small epsilon to catch exact match
                    improved = True
            elif self.monitor == 'val_loss':
                if best_val_loss - val_loss > self.min_delta:
                    improved = True
            
            if improved:
                patience_counter = 0
                # Save the best model weights (in memory)
                best_model_state = model.state_dict().copy()
                logger.debug(f'Model improved on {self.monitor}')
            else:
                patience_counter += 1
                logger.debug(f'Model did not improve for {patience_counter} epochs')
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
        
        # Calculate GPU time if CUDA is available
        gpu_time_ms = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_end_time.record()
            torch.cuda.synchronize()
            gpu_time_ms = gpu_start_time.elapsed_time(gpu_end_time)
            logger.info(f"GPU training time: {gpu_time_ms/1000:.3f} seconds")
        
        # Record model state in shared parameter pool if using weight sharing
        if self.enable_weight_sharing:
            self._update_shared_parameter_pool(model, complete_architecture, best_val_acc, best_val_loss)
            
        # Return evaluation results, including both best_val_acc and best_val_loss
        results = {
            'architecture': complete_architecture,
            'dataset': dataset_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,  # Ensure this is always included
            'test_acc': test_acc,
            'test_loss': test_loss,
            'epochs_trained': len(train_losses),
            'training_time': training_time,
            'gpu_time_ms': gpu_time_ms,
            'model_size': self._calculate_model_size(model),
            'flops_estimate': self._estimate_flops(model, dataset_config['input_shape']),
            'monitor_metric': self.monitor,  # Include which metric was monitored
            'used_shared_weights': used_shared_weights if self.enable_weight_sharing else False,
            'weight_sharing_enabled': self.enable_weight_sharing
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
            
            # Check for NaN in loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected in batch {batch_idx}, skipping backward pass")
                continue
                
            # Backward pass and optimize
            loss.backward()
            
            # Gradient value check and clipping to prevent exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if torch.isnan(grad_norm):
                logger.warning(f"NaN gradient detected in batch {batch_idx}, skipping optimizer step")
                # Zero out gradients with NaN values
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        param.grad.zero_()
            
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Accumulate loss (with NaN check)
            total_loss += loss.item() if not torch.isnan(loss) else 0
            
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
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    logger.warning("NaN loss detected during validation")
                    continue
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Accumulate loss (with NaN check)
                total_loss += loss.item() if not torch.isnan(loss) else 0
        
        # Calculate metrics (with NaN prevention)
        if total <= 0 or len(data_loader) <= 0:
            logger.warning("No valid batches during validation")
            avg_loss = float('nan')
            avg_acc = 0
        else:
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
        
    def _build_with_weight_sharing(self, architecture):
        """
        Build a model with weight sharing from the shared parameter pool.
        
        This implements the key idea from the ENAS paper (Pham et al., 2018) where
        parameters are shared between different architectures to avoid training from scratch.
        
        Args:
            architecture: Architecture specification dictionary
            
        Returns:
            tuple: (model, optimizer, criterion, used_shared_weights)
        """
        network_type = architecture.get('network_type', 'cnn')
        
        # First build the model normally to get the structure
        model, optimizer, criterion = self.model_builder.build_training_components(architecture)
        
        # Flag to track if we used shared weights
        used_shared_weights = False
        
        # If we have shared parameters for this network type, try to use them
        if network_type in self.shared_param_pools and self.shared_param_pools[network_type]:
            try:
                # Try to find the best matching weights from our pool
                compatible_weights = self._find_compatible_weights(model, architecture, network_type)
                
                if compatible_weights:
                    # Load the compatible weights
                    model, success = self._load_compatible_weights(model, compatible_weights)
                    
                    if success:
                        used_shared_weights = True
                        logger.info(f"Using shared weights for {network_type} architecture")
                        
                        # Reinitialize optimizer with the loaded model parameters
                        # This ensures the optimizer state is correctly matched to current parameters
                        optimizer_type = architecture.get('optimizer', 'adam').lower()
                        lr = architecture.get('learning_rate', 0.001)
                        
                        if optimizer_type == 'sgd':
                            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                        elif optimizer_type == 'adamw':
                            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                        else:  # default to adam
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            except Exception as e:
                # If any errors occur during weight sharing, log and continue without shared weights
                logger.warning(f"Error when trying to use shared weights: {str(e)}")
                # Rebuild the model from scratch
                model, optimizer, criterion = self.model_builder.build_training_components(architecture)
        
        return model, optimizer, criterion, used_shared_weights
    
    def _find_compatible_weights(self, model, architecture, network_type):
        """
        Find the most compatible weights in the shared parameter pool.
        
        Args:
            model: The model to find compatible weights for
            architecture: The architecture specification
            network_type: Type of network (cnn, mlp, etc.)
            
        Returns:
            dict or None: The most compatible weights or None if no compatible weights found
        """
        if not self.shared_param_pools[network_type]:
            return None
            
        # Extract key architecture features for matching
        num_layers = architecture.get('num_layers', 0)
        
        # Weight candidates by their similarity to current architecture
        candidates = []
        
        for entry in self.shared_param_pools[network_type]:
            shared_arch = entry['architecture']
            state_dict = entry['state_dict']
            performance = entry['performance']
            
            # Calculate similarity score based on architecture parameters
            similarity = self._calculate_architecture_similarity(architecture, shared_arch)
            
            # Higher similarity and better performance is better
            # This balances between similar structure and good performance
            if self.monitor == 'val_acc':
                score = similarity * (1.0 + performance)  # Higher acc is better
            else:
                # For loss, lower is better, so invert the performance component
                score = similarity * (1.0 + (1.0 / (performance + 1e-6)))
                
            candidates.append({
                'state_dict': state_dict,
                'similarity': similarity,
                'performance': performance,
                'score': score
            })
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the best candidate's state dict, or None if no candidates
        return candidates[0]['state_dict'] if candidates else None
    
    def _calculate_architecture_similarity(self, arch1, arch2):
        """
        Calculate a similarity score between two architectures.
        
        Args:
            arch1, arch2: Two architecture specifications to compare
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Start with a base similarity
        similarity = 0.0
        
        # Check if they're the same network type (most important)
        if arch1.get('network_type') != arch2.get('network_type'):
            return 0.0
            
        # Check number of layers
        layers1 = arch1.get('num_layers', 0)
        layers2 = arch2.get('num_layers', 0)
        
        # Layer count similarity (ratio of smaller to larger)
        if max(layers1, layers2) > 0:
            layer_similarity = min(layers1, layers2) / max(layers1, layers2)
        else:
            layer_similarity = 0.0
            
        # Layer structure similarity depends on network type
        network_type = arch1.get('network_type', '')
        structure_similarity = 0.0
        
        if network_type in ['cnn', 'resnet', 'mobilenet']:
            # Compare filters and kernel sizes
            if 'filters' in arch1 and 'filters' in arch2:
                min_layers = min(len(arch1['filters']), len(arch2['filters']))
                max_layers = max(len(arch1['filters']), len(arch2['filters']))
                
                # Count matching filters
                matches = sum(1 for i in range(min_layers) 
                             if arch1['filters'][i] == arch2['filters'][i])
                
                if max_layers > 0:
                    structure_similarity += 0.5 * (matches / max_layers)
            
            # Compare kernel sizes
            if 'kernel_sizes' in arch1 and 'kernel_sizes' in arch2:
                min_layers = min(len(arch1['kernel_sizes']), len(arch2['kernel_sizes']))
                max_layers = max(len(arch1['kernel_sizes']), len(arch2['kernel_sizes']))
                
                # Count matching kernel sizes
                matches = sum(1 for i in range(min_layers) 
                             if arch1['kernel_sizes'][i] == arch2['kernel_sizes'][i])
                
                if max_layers > 0:
                    structure_similarity += 0.25 * (matches / max_layers)
                    
        elif network_type in ['mlp', 'enhanced_mlp']:
            # Compare hidden units
            if 'hidden_units' in arch1 and 'hidden_units' in arch2:
                min_layers = min(len(arch1['hidden_units']), len(arch2['hidden_units']))
                max_layers = max(len(arch1['hidden_units']), len(arch2['hidden_units']))
                
                # For each layer, calculate similarity of hidden unit size ratio
                unit_similarities = []
                for i in range(min_layers):
                    units1 = arch1['hidden_units'][i]
                    units2 = arch2['hidden_units'][i]
                    unit_similarities.append(min(units1, units2) / max(units1, units2))
                
                if unit_similarities:
                    structure_similarity = sum(unit_similarities) / max_layers
        
        # Combine similarities (network type is most important, then layer count, then structure)
        similarity = 0.5 * layer_similarity + 0.5 * structure_similarity
        
        return similarity
    
    def _load_compatible_weights(self, model, state_dict):
        """
        Load compatible weights into a model, handling layer mismatches.
        
        This implements partial weight loading with shape checking to transfer
        as many parameters as possible between architectures.
        
        Args:
            model: PyTorch model to load weights into
            state_dict: State dictionary containing weights
            
        Returns:
            tuple: (updated_model, success_flag)
        """
        # Flag to track if we loaded any weights
        any_weights_loaded = False
        
        # Get the current model's state dict
        model_state = model.state_dict()
        
        # New state dict that we'll populate with compatible weights
        new_state = {}
        
        # For each parameter in the model
        for name, param in model_state.items():
            # If this parameter exists in the shared weights
            if name in state_dict:
                shared_param = state_dict[name]
                
                # Check if shapes match
                if param.shape == shared_param.shape:
                    # Use the shared parameter
                    new_state[name] = shared_param
                    any_weights_loaded = True
                else:
                    # Shapes don't match, use model's original parameter
                    new_state[name] = param
            else:
                # Parameter doesn't exist in shared weights, use original
                new_state[name] = param
                
        # Load the new state dict with combined parameters
        model.load_state_dict(new_state)
        
        return model, any_weights_loaded
    
    def estimate_performance_from_shared_weights(self, architecture, dataset_name, metric='val_acc'):
        """
        Estimate the performance of an architecture using the shared weights pool.
        This is used by the PNAS+ENAS combined approach to leverage weight sharing
        for fast performance estimation.
        
        Args:
            architecture: Architecture to estimate performance for
            dataset_name: Name of the dataset
            metric: The metric to use for performance estimation
            
        Returns:
            float or None: Estimated performance, or None if no good match found
        """
        if not self.enable_weight_sharing:
            return None
            
        network_type = architecture.get('network_type', 'cnn')
        
        # If no shared weights for this network type, return None
        if not network_type in self.shared_param_pools or not self.shared_param_pools[network_type]:
            return None
            
        # Find the most compatible weights
        try:
            # Build model to get structure
            model, _, _, _ = self._build_with_weight_sharing(architecture)
            
            # Get the top 3 candidates
            candidates = []
            
            for entry in self.shared_param_pools[network_type]:
                shared_arch = entry['architecture']
                
                # Get the appropriate performance value based on the metric
                if metric == 'val_acc':
                    idx = network_type in self.shared_weight_performance and len(self.shared_weight_performance[network_type]) > 0
                    if idx:
                        performance = self.shared_weight_performance[network_type][-1]['val_acc']
                    else:
                        performance = entry['performance'] if self.monitor == 'val_acc' else 0.0
                else:  # val_loss
                    idx = network_type in self.shared_weight_performance and len(self.shared_weight_performance[network_type]) > 0
                    if idx:
                        performance = self.shared_weight_performance[network_type][-1]['val_loss']
                    else:
                        performance = entry['performance'] if self.monitor == 'val_loss' else float('inf')
                
                # Calculate similarity score
                similarity = self._calculate_architecture_similarity(architecture, shared_arch)
                
                # Only include if similarity is above threshold
                if similarity > 0.7:  # Threshold for considering a match
                    candidates.append({
                        'similarity': similarity,
                        'performance': performance
                    })
                    
            # If no good candidates, return None
            if not candidates:
                return None
                
            # Sort by similarity
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Get top candidates (up to 3)
            top_candidates = candidates[:min(3, len(candidates))]
            
            # Calculate weighted average of performances based on similarity
            total_similarity = sum(c['similarity'] for c in top_candidates)
            weighted_performance = sum(c['similarity'] * c['performance'] for c in top_candidates) / total_similarity
            
            return weighted_performance
            
        except Exception as e:
            logger.warning(f"Error estimating performance from shared weights: {str(e)}")
            return None
    
    def _update_shared_parameter_pool(self, model, architecture, val_acc, val_loss):
        """
        Update the shared parameter pool with the current model's weights.
        
        Args:
            model: Trained PyTorch model
            architecture: Architecture specification
            val_acc: Validation accuracy
            val_loss: Validation loss
        """
        network_type = architecture.get('network_type', 'cnn')
        
        # Use the appropriate performance metric based on what we're monitoring
        performance = val_acc if self.monitor == 'val_acc' else val_loss
        
        # Create an entry for the parameter pool
        pool_entry = {
            'architecture': architecture.copy(),
            'state_dict': copy.deepcopy(model.state_dict()),
            'performance': performance,
            'network_type': network_type,  # Add network_type for easier access
            'val_acc': val_acc,  # Include both metrics for flexibility
            'val_loss': val_loss
        }
        
        # Add to the pool for this network type
        self.shared_param_pools[network_type].append(pool_entry)
        
        # Record performance for this set of weights
        self.shared_weight_performance[network_type].append({
            'val_acc': val_acc,
            'val_loss': val_loss
        })
        
        # Update the global weight_sharing_pool for PNAS+ENAS integration
        if hasattr(self, 'weight_sharing_pool') and self.weight_sharing_pool is not None:
            # Also add to the weight_sharing_pool for PNAS+ENAS combined search
            self.weight_sharing_pool.append(pool_entry)
            
            # Limit the size of the global pool
            if len(self.weight_sharing_pool) > self.weight_sharing_max_models:
                # Sort by performance (higher acc or lower loss is better)
                if self.monitor == 'val_acc':
                    # Sort by validation accuracy (higher is better)
                    self.weight_sharing_pool.sort(
                        key=lambda x: x['val_acc'], reverse=True
                    )
                else:
                    # Sort by validation loss (lower is better)
                    self.weight_sharing_pool.sort(
                        key=lambda x: x['val_loss'], reverse=False
                    )
                    
                # Keep only the top performers
                self.weight_sharing_pool = self.weight_sharing_pool[:self.weight_sharing_max_models]
                
            logger.info(f"Updated global weight sharing pool. Size: {len(self.weight_sharing_pool)}")
        
        # If the pool for this network type is getting too large, trim it
        if len(self.shared_param_pools[network_type]) > self.weight_sharing_max_models:
            # Sort by performance (higher acc or lower loss is better)
            if self.monitor == 'val_acc':
                # Sort by validation accuracy (higher is better)
                self.shared_param_pools[network_type].sort(
                    key=lambda x: x['performance'], reverse=True
                )
            else:
                # Sort by validation loss (lower is better)
                self.shared_param_pools[network_type].sort(
                    key=lambda x: x['performance'], reverse=False
                )
                
            # Keep only the top performers
            self.shared_param_pools[network_type] = \
                self.shared_param_pools[network_type][:self.weight_sharing_max_models]
            
        logger.info(f"Updated shared parameter pool for {network_type}. "
                   f"Pool size: {len(self.shared_param_pools[network_type])}")