"""
ENAS (Efficient Neural Architecture Search) Implementation

This module implements the ENAS algorithm as described in the paper:
"Efficient Neural Architecture Search via Parameter Sharing" by Pham et al. (2018)
https://arxiv.org/abs/1802.03268

ENAS introduces parameter sharing between different architectures to dramatically 
reduce the computational cost of architecture search.
"""

import time
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional, Union

from ..utils.exceptions import SNASException, EvaluationError
from ..utils.state_manager import SearchStateManager

logger = logging.getLogger(__name__)

class ENASController(nn.Module):
    """
    ENAS controller network for sampling architectures.
    
    The controller is typically an LSTM that predicts architecture components
    and is trained with reinforcement learning to maximize the expected reward.
    """
    
    def __init__(self, 
                 input_size=64,
                 hidden_size=100, 
                 num_layers=1,
                 network_type_options=None,
                 layer_count_options=None,
                 filter_options=None,
                 kernel_options=None,
                 hidden_unit_options=None,
                 activation_options=None,
                 entropy_weight=0.1,
                 device='cuda'):
        """
        Initialize the ENAS controller.
        
        Args:
            input_size: Size of input embedding
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            network_type_options: List of network types to choose from
            layer_count_options: List of possible layer counts
            filter_options: List of possible filter counts (for CNNs)
            kernel_options: List of possible kernel sizes (for CNNs)
            hidden_unit_options: List of possible hidden unit counts (for MLPs)
            activation_options: List of possible activation functions
            entropy_weight: Weight for the entropy loss term
            device: Device to use for computation
        """
        super(ENASController, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.entropy_weight = entropy_weight
        self.device = device
        
        # Define architecture options
        self.network_type_options = network_type_options or ['cnn', 'mlp', 'enhanced_mlp', 'resnet', 'mobilenet']
        self.layer_count_options = layer_count_options or [2, 3, 4, 5, 6, 8, 10]
        self.filter_options = filter_options or [16, 32, 64, 128, 256, 512]
        self.kernel_options = kernel_options or [1, 3, 5, 7]
        self.hidden_unit_options = hidden_unit_options or [64, 128, 256, 512, 1024, 2048]
        self.activation_options = activation_options or ['relu', 'leaky_relu', 'elu', 'selu', 'gelu']
        
        # Number of options for each architecture component
        self.num_network_types = len(self.network_type_options)
        self.num_layer_counts = len(self.layer_count_options)
        self.num_filters = len(self.filter_options)
        self.num_kernels = len(self.kernel_options)
        self.num_hidden_units = len(self.hidden_unit_options)
        self.num_activations = len(self.activation_options)
        
        # Controller LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Initial input embedding
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.input_size))
        
        # Prediction layers for different architecture components
        self.network_type_layer = nn.Linear(self.hidden_size, self.num_network_types)
        self.layer_count_layer = nn.Linear(self.hidden_size, self.num_layer_counts)
        
        # CNN-specific layers
        self.filter_layer = nn.Linear(self.hidden_size, self.num_filters)
        self.kernel_layer = nn.Linear(self.hidden_size, self.num_kernels)
        
        # MLP-specific layers
        self.hidden_units_layer = nn.Linear(self.hidden_size, self.num_hidden_units)
        
        # Common layers
        self.activation_layer = nn.Linear(self.hidden_size, self.num_activations)
        self.skip_connection_layer = nn.Linear(self.hidden_size, 2)  # Binary choice
        self.batch_norm_layer = nn.Linear(self.hidden_size, 2)  # Binary choice
        
        # Move to device
        self.to(self.device)
        
        # Tracking attributes
        self.rewards = []
        self.log_probs = []
        self.entropies = []
    
    def forward(self, batch_size=1):
        """
        Generate a batch of architectures.
        
        Args:
            batch_size: Number of architectures to generate
            
        Returns:
            list: Batch of architecture specifications
            list: Log probabilities for each architecture
            list: Entropies for each architecture
        """
        # Initialize hidden states and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        hidden = (h0, c0)
        
        # Initialize input with start token
        inputs = self.start_token.repeat(batch_size, 1, 1)
        
        # Sample network type
        network_type_logits, hidden = self._forward_lstm(inputs, hidden)
        network_type_probs = F.softmax(self.network_type_layer(network_type_logits), dim=-1)
        network_types, log_probs, entropies = self._sample_from_probs(network_type_probs)
        
        # Sample number of layers
        layer_count_logits, hidden = self._forward_lstm(inputs, hidden)
        layer_count_probs = F.softmax(self.layer_count_layer(layer_count_logits), dim=-1)
        layer_counts, layer_log_probs, layer_entropies = self._sample_from_probs(layer_count_probs)
        
        # Add log probs and entropies
        log_probs += layer_log_probs
        entropies += layer_entropies
        
        # Sample batch norm (global for the network)
        bn_logits, hidden = self._forward_lstm(inputs, hidden)
        bn_probs = F.softmax(self.batch_norm_layer(bn_logits), dim=-1)
        use_batch_norm, bn_log_probs, bn_entropies = self._sample_from_probs(bn_probs)
        
        # Add log probs and entropies
        log_probs += bn_log_probs
        entropies += bn_entropies
        
        # Create architecture specifications
        architectures = []
        for i in range(batch_size):
            net_type_idx = network_types[i]
            layer_count_idx = layer_counts[i]
            
            network_type = self.network_type_options[net_type_idx]
            num_layers = self.layer_count_options[layer_count_idx]
            
            # Initialize architecture
            architecture = {
                'network_type': network_type,
                'num_layers': num_layers,
                'use_batch_norm': bool(use_batch_norm[i])
            }
            
            # Sample layer-specific parameters
            if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
                # CNN architecture
                filters = []
                kernel_sizes = []
                activations = []
                use_skip_connections = []
                
                for _ in range(num_layers):
                    # Sample filter count
                    filter_logits, hidden = self._forward_lstm(inputs, hidden)
                    filter_probs = F.softmax(self.filter_layer(filter_logits), dim=-1)
                    filter_idx, filter_log_prob, filter_entropy = self._sample_from_probs(filter_probs)
                    filters.append(self.filter_options[filter_idx[i]])
                    log_probs += filter_log_prob
                    entropies += filter_entropy
                    
                    # Sample kernel size
                    kernel_logits, hidden = self._forward_lstm(inputs, hidden)
                    kernel_probs = F.softmax(self.kernel_layer(kernel_logits), dim=-1)
                    kernel_idx, kernel_log_prob, kernel_entropy = self._sample_from_probs(kernel_probs)
                    kernel_sizes.append(self.kernel_options[kernel_idx[i]])
                    log_probs += kernel_log_prob
                    entropies += kernel_entropy
                    
                    # Sample activation
                    activation_logits, hidden = self._forward_lstm(inputs, hidden)
                    activation_probs = F.softmax(self.activation_layer(activation_logits), dim=-1)
                    activation_idx, activation_log_prob, activation_entropy = self._sample_from_probs(activation_probs)
                    activations.append(self.activation_options[activation_idx[i]])
                    log_probs += activation_log_prob
                    entropies += activation_entropy
                    
                    # Sample skip connection (for resnet and some CNNs)
                    if network_type in ['resnet', 'cnn']:
                        skip_logits, hidden = self._forward_lstm(inputs, hidden)
                        skip_probs = F.softmax(self.skip_connection_layer(skip_logits), dim=-1)
                        skip_idx, skip_log_prob, skip_entropy = self._sample_from_probs(skip_probs)
                        use_skip_connections.append(bool(skip_idx[i]))
                        log_probs += skip_log_prob
                        entropies += skip_entropy
                
                # Add CNN-specific parameters
                architecture['filters'] = filters
                architecture['kernel_sizes'] = kernel_sizes
                architecture['activations'] = activations
                
                if network_type in ['resnet', 'cnn']:
                    architecture['use_skip_connections'] = use_skip_connections
            
            elif network_type in ['mlp', 'enhanced_mlp']:
                # MLP architecture
                hidden_units = []
                activations = []
                
                for _ in range(num_layers):
                    # Sample hidden units
                    hidden_unit_logits, hidden = self._forward_lstm(inputs, hidden)
                    hidden_unit_probs = F.softmax(self.hidden_units_layer(hidden_unit_logits), dim=-1)
                    hidden_unit_idx, hidden_unit_log_prob, hidden_unit_entropy = self._sample_from_probs(hidden_unit_probs)
                    hidden_units.append(self.hidden_unit_options[hidden_unit_idx[i]])
                    log_probs += hidden_unit_log_prob
                    entropies += hidden_unit_entropy
                    
                    # Sample activation
                    activation_logits, hidden = self._forward_lstm(inputs, hidden)
                    activation_probs = F.softmax(self.activation_layer(activation_logits), dim=-1)
                    activation_idx, activation_log_prob, activation_entropy = self._sample_from_probs(activation_probs)
                    activations.append(self.activation_options[activation_idx[i]])
                    log_probs += activation_log_prob
                    entropies += activation_entropy
                
                # Add MLP-specific parameters
                architecture['hidden_units'] = hidden_units
                architecture['activations'] = activations
                
                # Enhanced MLP-specific parameters
                if network_type == 'enhanced_mlp':
                    # Sample residual connections
                    residual_logits, hidden = self._forward_lstm(inputs, hidden)
                    residual_probs = F.softmax(self.skip_connection_layer(residual_logits), dim=-1)
                    residual_idx, residual_log_prob, residual_entropy = self._sample_from_probs(residual_probs)
                    architecture['use_residual'] = bool(residual_idx[i])
                    log_probs += residual_log_prob
                    entropies += residual_entropy
                    
                    # Sample layer normalization
                    layer_norm_logits, hidden = self._forward_lstm(inputs, hidden)
                    layer_norm_probs = F.softmax(self.skip_connection_layer(layer_norm_logits), dim=-1)
                    layer_norm_idx, layer_norm_log_prob, layer_norm_entropy = self._sample_from_probs(layer_norm_probs)
                    architecture['use_layer_norm'] = bool(layer_norm_idx[i])
                    log_probs += layer_norm_log_prob
                    entropies += layer_norm_entropy
            
            # Add to list of sampled architectures
            architectures.append(architecture)
        
        return architectures, log_probs, entropies
    
    def _forward_lstm(self, inputs, hidden):
        """Forward pass through the LSTM."""
        output, hidden = self.lstm(inputs, hidden)
        logits = output[:, -1, :]
        return logits, hidden
    
    def _sample_from_probs(self, probs):
        """Sample from probability distribution with entropy calculation."""
        # Calculate log probabilities and entropy
        log_probs = torch.log(probs + 1e-8)
        entropy = -(log_probs * probs).sum(dim=-1)
        
        # Sample from the distribution
        if self.training:
            samples = torch.multinomial(probs, 1).squeeze(-1)
        else:
            # During evaluation, take the most likely option
            samples = probs.argmax(dim=-1)
        
        # Get the log probabilities of the sampled actions
        selected_log_probs = log_probs.gather(1, samples.unsqueeze(1)).squeeze(1)
        
        return samples, [selected_log_probs], [entropy]
    
    def update_controller(self, rewards):
        """
        Update the controller using REINFORCE.
        
        Args:
            rewards: List of rewards for each sampled architecture
        """
        rewards = torch.tensor(rewards, device=self.device)
        
        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = 0
        for log_prob, reward in zip(self.log_probs, rewards):
            policy_loss -= log_prob * reward
        
        # Calculate entropy loss (to encourage exploration)
        entropy_loss = 0
        for entropy in self.entropies:
            entropy_loss -= entropy
        
        # Total loss
        loss = policy_loss + self.entropy_weight * entropy_loss
        
        # Update controller
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clear saved rewards and log probs
        self.rewards = []
        self.log_probs = []
        self.entropies = []
        
        return loss.item()


class ENASSearch:
    """
    Efficient Neural Architecture Search implementation.
    
    ENAS uses a controller to sample architectures and parameter sharing
    to enable efficient evaluation of many architectures.
    """
    
    def __init__(self, architecture_space, evaluator, dataset_name,
                controller_sample_count=50, controller_entropy_weight=0.1,
                weight_sharing_max_models=200, num_iterations=10,
                metric='val_acc', checkpoint_frequency=0,
                output_dir="output", results_dir="output/results"):
        """
        Initialize the ENAS search.
        
        Args:
            architecture_space: Space of possible architectures
            evaluator: Component to evaluate architectures
            dataset_name: Name of the dataset to use
            controller_sample_count: Number of architectures to sample per iteration
            controller_entropy_weight: Weight for entropy term in controller update
            weight_sharing_max_models: Maximum size of weight sharing pool
            num_iterations: Number of search iterations
            metric: Metric to optimize ('val_acc', 'val_loss', etc.)
            checkpoint_frequency: Save checkpoint every N iterations (0 to disable)
            output_dir: Directory to store output files
            results_dir: Directory to store result files
        """
        self.architecture_space = architecture_space
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.controller_sample_count = controller_sample_count
        self.controller_entropy_weight = controller_entropy_weight
        self.weight_sharing_max_models = weight_sharing_max_models
        self.num_iterations = num_iterations
        self.metric = metric
        self.checkpoint_frequency = checkpoint_frequency
        
        # Make sure weight sharing is enabled in the evaluator
        self.evaluator.enable_weight_sharing = True
        self.evaluator.weight_sharing_max_models = weight_sharing_max_models
        
        # Set up state manager for checkpointing
        self.state_manager = SearchStateManager(output_dir, results_dir)
        
        # Set up fitness comparison based on metric type
        # For accuracy metrics, higher is better; for loss metrics, lower is better
        self.higher_is_better = not metric.endswith('loss')
        
        # Initialize with worst possible values based on optimization direction
        if self.higher_is_better:
            self.best_fitness = float('-inf')
        else:
            self.best_fitness = float('inf')
            
        self.best_architecture = None
        
        # Initialize controller
        device = evaluator.device
        self.controller = ENASController(
            entropy_weight=controller_entropy_weight,
            device=device
        )
        
        # For tracking all evaluated architectures
        self.evaluated_architectures = {}
        
        # Initialize history
        self.history = {
            'iterations': [],
            'best_fitness': [],
            'best_architecture': [],
            'controller_loss': [],
            'evaluation_times': [],
            'sampled_architectures': [],
            'metric': metric,
            'metric_type': 'loss' if metric.endswith('loss') else 'accuracy'
        }
        
        logger.info(f"ENAS search initialized with metric: {metric} "
                   f"({'higher is better' if self.higher_is_better else 'lower is better'})")
    
    def _evaluate_architecture(self, architecture, fast_mode=False):
        """
        Evaluate a single architecture with parameter sharing.
        
        Args:
            architecture: Architecture to evaluate
            fast_mode: Whether to use fast evaluation
            
        Returns:
            float: Performance value
        """
        # Check if we've evaluated this architecture before
        arch_str = json.dumps(architecture, sort_keys=True)
        if arch_str in self.evaluated_architectures:
            # Use cached result
            evaluation = self.evaluated_architectures[arch_str]
            return self._get_fitness_from_evaluation(evaluation)
        
        # Evaluate the architecture
        try:
            evaluation = self.evaluator.evaluate(
                architecture, self.dataset_name, fast_mode=fast_mode
            )
            # Cache the result
            self.evaluated_architectures[arch_str] = evaluation
            
            # Return fitness
            return self._get_fitness_from_evaluation(evaluation)
        except Exception as e:
            logger.error(f"Error evaluating architecture: {e}")
            # Return worst fitness
            if self.higher_is_better:
                return float('-inf')
            else:
                return float('inf')
    
    def _get_fitness_from_evaluation(self, evaluation):
        """
        Extract fitness value from evaluation result based on metric.
        
        Args:
            evaluation: Result dictionary from evaluator
            
        Returns:
            float: Fitness value
        """
        # If monitoring validation loss, use best_val_loss (lowest seen)
        # If monitoring validation accuracy, use best_val_acc (highest seen)
        if self.metric == 'val_acc':
            fitness = evaluation.get('best_val_acc', 0.0)
        elif self.metric == 'val_loss':
            fitness = evaluation.get('best_val_loss', float('inf'))
        elif self.metric == 'test_acc':
            fitness = evaluation.get('test_acc', 0.0)
        elif self.metric == 'test_loss':
            fitness = evaluation.get('test_loss', float('inf'))
        else:
            # For any other metric, try to get it directly
            fitness = evaluation.get(self.metric, 0.0 if self.higher_is_better else float('inf'))
        
        return fitness
    
    def get_checkpoint_state(self, iteration):
        """
        Get the current state for checkpointing.
        
        Args:
            iteration: Current iteration
            
        Returns:
            dict: Current search state
        """
        checkpoint_state = {
            'evaluated_architectures': self.evaluated_architectures,
            'higher_is_better': self.higher_is_better,
            'best_fitness': self.best_fitness,
            'best_architecture': self.best_architecture,
            'architecture_space_state': self.architecture_space.__dict__,
            'controller_state': self.controller.state_dict(),
            'search_params': {
                'controller_sample_count': self.controller_sample_count,
                'controller_entropy_weight': self.controller_entropy_weight,
                'weight_sharing_max_models': self.weight_sharing_max_models,
                'num_iterations': self.num_iterations,
                'metric': self.metric
            },
            'iteration': iteration
        }
        
        return checkpoint_state
    
    def restore_from_checkpoint(self, checkpoint):
        """
        Restore search state from a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            
        Returns:
            int: Iteration to resume from
        """
        try:
            # Restore search state
            search_state = checkpoint['search_state']
            
            # Restore evaluated architectures
            self.evaluated_architectures = search_state.get('evaluated_architectures', {})
            
            # Restore best architecture and fitness
            self.best_architecture = checkpoint['best_architecture']
            self.best_fitness = checkpoint['best_fitness']
            
            # Restore controller state
            if 'controller_state' in search_state:
                self.controller.load_state_dict(search_state['controller_state'])
            
            # Restore history
            self.history = checkpoint['history']
            
            # Get the iteration to resume from
            iteration = search_state['iteration']
            
            logger.info(f"Restored search state from checkpoint (iteration {iteration})")
            return iteration
            
        except KeyError as e:
            raise SNASException(f"Invalid checkpoint format: missing key {e}")
    
    def search(self, resume_from=None, progress_callback=None, stop_condition=None):
        """
        Run the ENAS search process.
        
        Args:
            resume_from: Path to checkpoint file to resume from
            progress_callback: Optional callback function to report progress (0.0-1.0)
            stop_condition: Optional function that returns True if search should be stopped
            
        Returns:
            dict: Best architecture found
            float: Best fitness value
            dict: Search history
        """
        logger.info("Starting ENAS search")
        
        # Handle resuming from checkpoint
        start_iteration = 0
        if resume_from:
            try:
                checkpoint = self.state_manager.load_checkpoint(resume_from)
                start_iteration = self.restore_from_checkpoint(checkpoint)
                # Start from the next iteration
                start_iteration += 1
                logger.info(f"Resuming search from iteration {start_iteration}")
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                logger.info("Starting new search instead")
                start_iteration = 0
        
        # Report initial progress
        if progress_callback:
            progress_callback(0.0, f"Initializing ENAS controller")
            
        # Main search loop
        for iteration in range(start_iteration, self.num_iterations):
            # Check if search should be stopped
            if stop_condition and stop_condition():
                logger.info(f"Search stopped at iteration {iteration}/{self.num_iterations}")
                if progress_callback:
                    progress_callback(
                        (iteration - start_iteration) / (self.num_iterations - start_iteration),
                        f"Search stopped at iteration {iteration}/{self.num_iterations}"
                    )
                break
                
            logger.info(f"Iteration {iteration+1}/{self.num_iterations}")
            
            # Report progress to UI
            if progress_callback:
                # Calculate progress based on iteration
                progress_value = (iteration - start_iteration) / (self.num_iterations - start_iteration)
                progress_callback(progress_value, f"Iteration {iteration+1}/{self.num_iterations}")
            
            # Sample architectures from controller
            self.controller.train()
            sampled_architectures, log_probs, entropies = self.controller.forward(batch_size=self.controller_sample_count)
            
            # Save for history
            self.history['sampled_architectures'].append(sampled_architectures)
            
            # Report sampling completion
            if progress_callback:
                progress_callback(
                    progress_value, 
                    f"Sampled {len(sampled_architectures)} architectures in iteration {iteration+1}"
                )
            
            # Evaluate architectures
            rewards = []
            performances = []
            
            start_time = time.time()
            for i, arch in enumerate(sampled_architectures):
                # Update progress during architecture evaluation
                if progress_callback and i % max(1, len(sampled_architectures)//5) == 0:
                    subprogress = progress_value + ((i / len(sampled_architectures)) * (1/self.num_iterations))
                    progress_callback(
                        min(subprogress, 1.0),  # Ensure we don't exceed 1.0
                        f"Evaluating architecture {i+1}/{len(sampled_architectures)} in iteration {iteration+1}"
                    )
                
                # Use fast mode for early iterations
                use_fast_mode = iteration < self.num_iterations // 3
                
                # Evaluate the architecture
                performance = self._evaluate_architecture(arch, fast_mode=use_fast_mode)
                performances.append(performance)
                
                # Convert performance to reward (higher is always better for the controller)
                reward = performance if self.higher_is_better else -performance
                rewards.append(reward)
                
                # Update best architecture if better
                if (self.higher_is_better and performance > self.best_fitness) or \
                   (not self.higher_is_better and performance < self.best_fitness):
                    self.best_fitness = performance
                    self.best_architecture = arch.copy()
                    logger.info(f"New best architecture found! Fitness: {performance:.4f}")
                    
                    if progress_callback:
                        progress_callback(
                            min(subprogress, 1.0),
                            f"New best architecture found! Fitness: {performance:.4f}"
                        )
            
            # Calculate evaluation time
            eval_time = time.time() - start_time
            
            # Report controller update
            if progress_callback:
                progress_callback(
                    progress_value,
                    f"Updating ENAS controller in iteration {iteration+1}"
                )
            
            # Update the controller
            controller_loss = self.controller.update_controller(rewards)
            
            # Save to history
            self.history['iterations'].append(iteration)
            self.history['best_fitness'].append(self.best_fitness)
            self.history['best_architecture'].append(self.best_architecture.copy() if self.best_architecture else None)
            self.history['controller_loss'].append(controller_loss)
            self.history['evaluation_times'].append(eval_time)
            
            # Log progress
            logger.info(f"Iteration {iteration+1} completed. Best fitness: {self.best_fitness:.4f}, "
                       f"Controller loss: {controller_loss:.4f}, Evaluation time: {eval_time:.2f}s")
                       
            # Report iteration completion
            if progress_callback:
                progress_callback(
                    (iteration - start_iteration + 1) / (self.num_iterations - start_iteration),
                    f"Completed iteration {iteration+1}/{self.num_iterations}. Best: {self.best_fitness:.4f}"
                )
            
            # Save checkpoint if needed
            if self.checkpoint_frequency > 0 and (iteration + 1) % self.checkpoint_frequency == 0:
                try:
                    # Get checkpoint state
                    checkpoint_state = self.get_checkpoint_state(iteration)
                    
                    # Save checkpoint
                    self.state_manager.save_checkpoint(
                        dataset_name=self.dataset_name,
                        search_state=checkpoint_state,
                        best_architecture=self.best_architecture,
                        best_fitness=self.best_fitness,
                        history=self.history,
                        generation=iteration  # Use iteration as generation
                    )
                    logger.info(f"Checkpoint saved at iteration {iteration+1}")
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
        
        # Final evaluation of the best architecture
        logger.info("Performing final evaluation of best architecture")
        try:
            evaluation = self.evaluator.evaluate(
                self.best_architecture, self.dataset_name, fast_mode=False
            )
            
            # Update best fitness with full evaluation result
            self.best_fitness = self._get_fitness_from_evaluation(evaluation)
            logger.info(f"Final evaluation result: {self.best_fitness:.4f}")
        except Exception as e:
            logger.error(f"Error in final evaluation: {e}")
        
        logger.info("ENAS search completed")
        return self.best_architecture, self.best_fitness, self.history