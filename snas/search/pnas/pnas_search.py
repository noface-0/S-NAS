"""
PNAS (Progressive Neural Architecture Search) Implementation

This module implements the PNAS algorithm as described in the paper:
"Progressive Neural Architecture Search" by Liu et al. (2018)
https://arxiv.org/abs/1712.00559

PNAS uses a surrogate model to predict architecture performance and a progressive
widening strategy to efficiently explore the architecture space.
"""

import time
import json
import logging
import numpy as np
import heapq
import torch
import random
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict

from .surrogate_model import SurrogateModel
from ...utils.exceptions import SNASException, EvaluationError
from ...utils.state_manager import SearchStateManager

logger = logging.getLogger(__name__)

class PNASSearch:
    """
    Progressive Neural Architecture Search algorithm.
    
    This class implements the PNAS algorithm with a surrogate predictor model
    to efficiently search for optimal neural network architectures.
    """
    
    def __init__(self, architecture_space, evaluator, dataset_name,
                 beam_size=10, num_expansions=5, max_complexity=3,
                 surrogate_model=None, predictor_batch_size=100,
                 metric='val_acc', checkpoint_frequency=0,
                 output_dir="output", results_dir="output/results",
                 use_shared_weights=False, shared_weights_importance=0.5):
        """
        Initialize the PNAS search.
        
        Args:
            architecture_space: Space of possible architectures
            evaluator: Component to evaluate architectures
            dataset_name: Name of the dataset to use
            beam_size: Number of architectures to keep in the beam
            num_expansions: Number of expansions per architecture in the beam
            max_complexity: Maximum complexity level to explore
            surrogate_model: Pre-trained surrogate model (or None to create new)
            predictor_batch_size: Batch size for surrogate model predictions
            metric: Metric to optimize ('val_acc', 'val_loss', etc.)
            checkpoint_frequency: Save checkpoint every N complexity levels (0 to disable)
            output_dir: Directory to store output files
            results_dir: Directory to store result files
            use_shared_weights: Whether to use ENAS-style shared weights for prediction (PNAS+ENAS hybrid)
            shared_weights_importance: Balance between surrogate model and shared weights predictions (0-1)
        """
        self.architecture_space = architecture_space
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.beam_size = beam_size
        self.num_expansions = num_expansions
        self.max_complexity = max_complexity
        self.predictor_batch_size = predictor_batch_size
        self.metric = metric
        self.checkpoint_frequency = checkpoint_frequency
        
        # PNAS+ENAS hybrid parameters
        self.use_shared_weights = use_shared_weights
        self.shared_weights_importance = shared_weights_importance
        
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
        
        # Initialize surrogate model
        if surrogate_model is None:
            device = evaluator.device
            self.surrogate_model = SurrogateModel(
                input_size=32,  # Will be adjusted dynamically in encode_architecture
                hidden_size=64,
                num_layers=2,
                dropout=0.1,
                device=device
            )
        else:
            self.surrogate_model = surrogate_model
            
        # Initialize beam with empty list (will be populated during search)
        self.beam = []
        
        # For tracking all evaluated architectures
        self.evaluated_architectures = {}
        
        # Initialize history
        self.history = {
            'complexity_levels': [],
            'beam_architectures': [],
            'beam_performances': [],
            'surrogate_train_losses': [],
            'surrogate_val_losses': [],
            'evaluation_times': [],
            'best_architecture': [],
            'best_fitness': [],
            'metric': metric,
            'metric_type': 'loss' if metric.endswith('loss') else 'accuracy'
        }
        
        logger.info(f"PNAS search initialized with metric: {metric} "
                   f"({'higher is better' if self.higher_is_better else 'lower is better'})")
                   
    def _sample_architecture_with_complexity(self, complexity_level):
        """
        Sample an architecture with the specified complexity level.
        
        Args:
            complexity_level: Current complexity level (1=simplest, 3=most complex)
            
        Returns:
            dict: Architecture with appropriate complexity
        """
        # Get a random architecture first
        architecture = self.architecture_space.sample_random_architecture()
        
        # Adjust complexity based on level
        if complexity_level == 1:
            # Simplest architectures - fewer layers, simpler components
            architecture['num_layers'] = min(3, architecture['num_layers'])
            
            # Simplify network type - avoid complex types at level 1
            simple_types = ['cnn', 'mlp']
            if architecture['network_type'] not in simple_types:
                architecture['network_type'] = random.choice(simple_types)
                
            # Reduce filters for CNNs or hidden units for MLPs
            if architecture['network_type'] == 'cnn':
                # Initialize CNN-specific parameters if they don't exist
                if 'filters' not in architecture:
                    architecture['filters'] = [64 for _ in range(architecture['num_layers'])]
                else:
                    architecture['filters'] = [min(f, 128) for f in architecture['filters']]
                
                if 'kernel_sizes' not in architecture:
                    architecture['kernel_sizes'] = [3 for _ in range(architecture['num_layers'])]
                
                if 'use_skip_connections' not in architecture:
                    architecture['use_skip_connections'] = [False for _ in range(architecture['num_layers'])]
                elif not isinstance(architecture['use_skip_connections'], list):
                    # Convert bool to list of bools if needed
                    architecture['use_skip_connections'] = [architecture['use_skip_connections']] * architecture['num_layers']
                
                architecture['use_batch_norm'] = False
                
            elif architecture['network_type'] == 'mlp':
                # Initialize hidden_units if it doesn't exist
                if 'hidden_units' not in architecture:
                    architecture['hidden_units'] = [512 for _ in range(architecture['num_layers'])]
                else:
                    architecture['hidden_units'] = [min(h, 512) for h in architecture['hidden_units']]
            
            # Simplify activation functions - just use ReLU at level 1
            architecture['activations'] = ['relu' for _ in range(architecture['num_layers'])]
                
        elif complexity_level == 2:
            # Medium complexity - moderate layers, some advanced features
            architecture['num_layers'] = min(5, architecture['num_layers'])
            
            # Allow more network types at level 2
            medium_types = ['cnn', 'mlp', 'enhanced_mlp', 'resnet', 'mobilenet']
            if architecture['network_type'] not in medium_types:
                architecture['network_type'] = random.choice(medium_types)
            
            # Some skip connections and batch norm allowed
            architecture['use_batch_norm'] = random.choice([True, False])
        
        # Handle special parameter types
        if 'use_skip_connections' in architecture and not isinstance(architecture['use_skip_connections'], list):
            architecture['use_skip_connections'] = [architecture['use_skip_connections']] * architecture['num_layers']
        
        return architecture
        
    def _expand_architecture(self, architecture):
        """
        Generate variations of an architecture by modifying its components.
        
        Args:
            architecture: Base architecture to expand
            
        Returns:
            list: List of expanded architecture variations
        """
        expanded = []
        network_type = architecture.get('network_type', 'cnn')
        
        # 1. Increase the number of layers
        if architecture['num_layers'] < 20:  # Limit max layers
            arch_more_layers = architecture.copy()
            arch_more_layers['num_layers'] = architecture['num_layers'] + 1
            
            # Add layer-specific parameters
            if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
                arch_more_layers['filters'] = architecture['filters'] + [architecture['filters'][-1]]
                arch_more_layers['kernel_sizes'] = architecture['kernel_sizes'] + [architecture['kernel_sizes'][-1]]
                arch_more_layers['activations'] = architecture['activations'] + [architecture['activations'][-1]]
                if 'use_skip_connections' in architecture:
                    if isinstance(architecture['use_skip_connections'], list):
                        arch_more_layers['use_skip_connections'] = architecture['use_skip_connections'] + [architecture['use_skip_connections'][-1]]
                    else:
                        arch_more_layers['use_skip_connections'] = [architecture['use_skip_connections']] * (architecture['num_layers'] + 1)
            
            elif network_type in ['mlp', 'enhanced_mlp']:
                arch_more_layers['hidden_units'] = architecture['hidden_units'] + [architecture['hidden_units'][-1]]
                arch_more_layers['activations'] = architecture['activations'] + [architecture['activations'][-1]]
            
            expanded.append(arch_more_layers)
        
        # 2. Modify filters/hidden units (increase/decrease)
        if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
            filter_options = [16, 32, 64, 128, 256, 512]
            for i in range(len(architecture['filters'])):
                current_filter = architecture['filters'][i]
                idx = filter_options.index(current_filter) if current_filter in filter_options else -1
                
                if idx > 0:  # Can decrease filters
                    arch_less_filters = architecture.copy()
                    arch_less_filters['filters'] = architecture['filters'].copy()
                    arch_less_filters['filters'][i] = filter_options[idx-1]
                    expanded.append(arch_less_filters)
                
                if idx < len(filter_options) - 1:  # Can increase filters
                    arch_more_filters = architecture.copy()
                    arch_more_filters['filters'] = architecture['filters'].copy()
                    arch_more_filters['filters'][i] = filter_options[idx+1]
                    expanded.append(arch_more_filters)
        
        elif network_type in ['mlp', 'enhanced_mlp']:
            hidden_options = [64, 128, 256, 512, 1024, 2048]
            for i in range(len(architecture['hidden_units'])):
                current_hidden = architecture['hidden_units'][i]
                idx = hidden_options.index(current_hidden) if current_hidden in hidden_options else -1
                
                if idx > 0:  # Can decrease hidden units
                    arch_less_hidden = architecture.copy()
                    arch_less_hidden['hidden_units'] = architecture['hidden_units'].copy()
                    arch_less_hidden['hidden_units'][i] = hidden_options[idx-1]
                    expanded.append(arch_less_hidden)
                
                if idx < len(hidden_options) - 1:  # Can increase hidden units
                    arch_more_hidden = architecture.copy()
                    arch_more_hidden['hidden_units'] = architecture['hidden_units'].copy()
                    arch_more_hidden['hidden_units'][i] = hidden_options[idx+1]
                    expanded.append(arch_more_hidden)
        
        # 3. Toggle batch normalization
        arch_toggle_bn = architecture.copy()
        arch_toggle_bn['use_batch_norm'] = not architecture.get('use_batch_norm', False)
        expanded.append(arch_toggle_bn)
        
        # 4. Change activation functions
        activation_options = ['relu', 'leaky_relu', 'elu', 'selu', 'gelu']
        for i in range(len(architecture['activations'])):
            for act in activation_options:
                if act != architecture['activations'][i]:
                    arch_diff_act = architecture.copy()
                    arch_diff_act['activations'] = architecture['activations'].copy()
                    arch_diff_act['activations'][i] = act
                    expanded.append(arch_diff_act)
        
        # 5. For CNN models, toggle skip connections
        if network_type in ['cnn', 'resnet'] and 'use_skip_connections' in architecture:
            if isinstance(architecture['use_skip_connections'], list):
                for i in range(len(architecture['use_skip_connections'])):
                    arch_toggle_skip = architecture.copy()
                    arch_toggle_skip['use_skip_connections'] = architecture['use_skip_connections'].copy()
                    arch_toggle_skip['use_skip_connections'][i] = not architecture['use_skip_connections'][i]
                    expanded.append(arch_toggle_skip)
            else:
                arch_toggle_skip = architecture.copy()
                arch_toggle_skip['use_skip_connections'] = not architecture['use_skip_connections']
                expanded.append(arch_toggle_skip)
        
        # 6. For MLP models, toggle residual/layer norm
        if network_type == 'enhanced_mlp':
            # Toggle residual
            arch_toggle_residual = architecture.copy()
            arch_toggle_residual['use_residual'] = not architecture.get('use_residual', False)
            expanded.append(arch_toggle_residual)
            
            # Toggle layer norm
            arch_toggle_ln = architecture.copy()
            arch_toggle_ln['use_layer_norm'] = not architecture.get('use_layer_norm', False)
            expanded.append(arch_toggle_ln)
        
        # 7. Randomly modify learning rate
        lr_options = [0.1, 0.01, 0.001, 0.0001]
        current_lr = architecture.get('learning_rate', 0.001)
        for lr in lr_options:
            if lr != current_lr:
                arch_diff_lr = architecture.copy()
                arch_diff_lr['learning_rate'] = lr
                expanded.append(arch_diff_lr)
        
        # 8. Change optimizer
        optimizer_options = ['sgd', 'adam', 'adamw', 'rmsprop']
        current_opt = architecture.get('optimizer', 'adam')
        for opt in optimizer_options:
            if opt != current_opt:
                arch_diff_opt = architecture.copy()
                arch_diff_opt['optimizer'] = opt
                expanded.append(arch_diff_opt)
        
        # Only return valid architectures
        valid_expansions = []
        for arch in expanded:
            if self.architecture_space.validate_architecture(arch):
                valid_expansions.append(arch)
        
        # Shuffle and return top 'num_expansions' architectures
        import random
        random.shuffle(valid_expansions)
        return valid_expansions[:self.num_expansions]
    
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

    def _evaluate_architectures(self, architectures, fast_mode=False):
        """
        Evaluate a list of architectures to get their performance.
        
        Args:
            architectures: List of architectures to evaluate
            fast_mode: Whether to use fast mode for evaluation
            
        Returns:
            list: Performance values for each architecture
        """
        performance_values = []
        
        for i, architecture in enumerate(architectures):
            # Check if we've evaluated this architecture before
            arch_str = json.dumps(architecture, sort_keys=True)
            if arch_str in self.evaluated_architectures:
                # Use cached result
                evaluation = self.evaluated_architectures[arch_str]
                logger.info(f"Architecture {i+1}/{len(architectures)} already evaluated, using cached result")
            else:
                # Evaluate the architecture
                logger.info(f"Evaluating architecture {i+1}/{len(architectures)}")
                try:
                    evaluation = self.evaluator.evaluate(
                        architecture, self.dataset_name, fast_mode=fast_mode
                    )
                    # Cache the result
                    self.evaluated_architectures[arch_str] = evaluation
                except EvaluationError as e:
                    logger.error(f"Architecture evaluation error: {e}")
                    # Assign a very poor fitness
                    if self.higher_is_better:
                        performance_values.append(float('-inf'))
                    else:
                        performance_values.append(float('inf'))
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error evaluating architecture: {e}")
                    # Assign a very poor fitness
                    if self.higher_is_better:
                        performance_values.append(float('-inf'))
                    else:
                        performance_values.append(float('inf'))
                    continue
            
            # Extract fitness from evaluation
            fitness = self._get_fitness_from_evaluation(evaluation)
            performance_values.append(fitness)
            
            # Update best architecture if better
            if (self.higher_is_better and fitness > self.best_fitness) or \
               (not self.higher_is_better and fitness < self.best_fitness):
                self.best_fitness = fitness
                self.best_architecture = architecture.copy()
                logger.info(f"New best architecture found! Fitness: {fitness:.4f}")
        
        return performance_values
        
    def _evaluate_architectures_with_progress(self, architectures, fast_mode=False, 
                                             progress_callback=None, current_progress_value=0.0,
                                             complexity_level=0):
        """
        Evaluate a list of architectures with UI progress reporting.
        
        Args:
            architectures: List of architectures to evaluate
            fast_mode: Whether to use fast mode for evaluation
            progress_callback: Function to report progress to the UI
            current_progress_value: Current overall progress value (0-1)
            complexity_level: Current complexity level
            
        Returns:
            list: Performance values for each architecture
        """
        performance_values = []
        total_archs = len(architectures)
        
        for i, architecture in enumerate(architectures):
            # Update sub-progress
            if progress_callback:
                sub_progress = i / total_archs if total_archs > 0 else 0
                progress_callback(
                    current_progress_value,
                    f"Complexity level {complexity_level}/{self.max_complexity}",
                    sub_progress,
                    f"Evaluating architecture {i+1}/{total_archs}"
                )
            
            # Check if we've evaluated this architecture before
            arch_str = json.dumps(architecture, sort_keys=True)
            if arch_str in self.evaluated_architectures:
                # Use cached result
                evaluation = self.evaluated_architectures[arch_str]
                logger.info(f"Architecture {i+1}/{total_archs} already evaluated, using cached result")
                if progress_callback:
                    progress_callback(
                        current_progress_value,
                        f"Complexity level {complexity_level}/{self.max_complexity}",
                        sub_progress,
                        f"Using cached result for architecture {i+1}/{total_archs}"
                    )
            else:
                # Evaluate the architecture
                logger.info(f"Evaluating architecture {i+1}/{total_archs}")
                try:
                    if progress_callback:
                        progress_callback(
                            current_progress_value,
                            f"Complexity level {complexity_level}/{self.max_complexity}",
                            sub_progress,
                            f"Training architecture {i+1}/{total_archs}"
                        )
                    
                    evaluation = self.evaluator.evaluate(
                        architecture, self.dataset_name, fast_mode=fast_mode
                    )
                    # Cache the result
                    self.evaluated_architectures[arch_str] = evaluation
                except EvaluationError as e:
                    logger.error(f"Architecture evaluation error: {e}")
                    # Assign a very poor fitness
                    if self.higher_is_better:
                        performance_values.append(float('-inf'))
                    else:
                        performance_values.append(float('inf'))
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error evaluating architecture: {e}")
                    # Assign a very poor fitness
                    if self.higher_is_better:
                        performance_values.append(float('-inf'))
                    else:
                        performance_values.append(float('inf'))
                    continue
            
            # Extract fitness from evaluation
            fitness = self._get_fitness_from_evaluation(evaluation)
            performance_values.append(fitness)
            
            # Update best architecture if better
            if (self.higher_is_better and fitness > self.best_fitness) or \
               (not self.higher_is_better and fitness < self.best_fitness):
                self.best_fitness = fitness
                self.best_architecture = architecture.copy()
                logger.info(f"New best architecture found! Fitness: {fitness:.4f}")
                if progress_callback:
                    progress_callback(
                        current_progress_value,
                        f"Complexity level {complexity_level}/{self.max_complexity}",
                        sub_progress,
                        f"New best architecture found! Fitness: {fitness:.4f}"
                    )
        
        # Mark evaluation as complete
        if progress_callback:
            progress_callback(
                current_progress_value,
                f"Complexity level {complexity_level}/{self.max_complexity}",
                1.0,  # Complete
                f"Evaluated {total_archs} architectures"
            )
        
        return performance_values
    
    def _predict_performance(self, architectures):
        """
        Use surrogate model to predict performance of architectures.
        For PNAS+ENAS hybrid mode, combines surrogate model predictions with
        shared weights evaluations.
        
        Args:
            architectures: List of architectures to predict
            
        Returns:
            list: Predicted performance values
        """
        # Check if we should use the hybrid PNAS+ENAS approach
        if self.use_shared_weights and hasattr(self.evaluator, 'weight_sharing_pool') and self.evaluator.weight_sharing_pool:
            # Get predictions from both surrogate model and shared weights
            surrogate_predictions = self._get_surrogate_predictions(architectures)
            shared_weights_predictions = self._get_shared_weights_predictions(architectures)
            
            # Make sure both prediction lists have the same length
            if len(surrogate_predictions) != len(shared_weights_predictions):
                logger.warning(f"Prediction length mismatch: surrogate={len(surrogate_predictions)}, shared_weights={len(shared_weights_predictions)}")
                # Use the smaller length to avoid index errors
                min_length = min(len(surrogate_predictions), len(shared_weights_predictions))
                surrogate_predictions = surrogate_predictions[:min_length]
                shared_weights_predictions = shared_weights_predictions[:min_length]
            
            # Combine predictions using weighted average
            combined_predictions = []
            for i in range(len(surrogate_predictions)):
                # Make sure predictions are numeric values, not None or tensors
                surrogate_pred = surrogate_predictions[i]
                shared_pred = shared_weights_predictions[i]
                
                # Convert to float values if possible
                if isinstance(surrogate_pred, torch.Tensor):
                    surrogate_pred = surrogate_pred.item()
                if isinstance(shared_pred, torch.Tensor):
                    shared_pred = shared_pred.item()
                
                # Handle None values
                if surrogate_pred is None and shared_pred is None:
                    # If both are None, use a default value
                    if self.higher_is_better:
                        combined_predictions.append(0.0)  # Default low accuracy
                    else:
                        combined_predictions.append(1.0)  # Default reasonable loss
                elif surrogate_pred is None:
                    combined_predictions.append(shared_pred)
                elif shared_pred is None:
                    combined_predictions.append(surrogate_pred)
                else:
                    # Weighted average of both predictions
                    combined = (
                        (1 - self.shared_weights_importance) * surrogate_pred +
                        self.shared_weights_importance * shared_pred
                    )
                    combined_predictions.append(combined)
            
            return combined_predictions
        else:
            # Standard PNAS: just use surrogate model predictions
            return self._get_surrogate_predictions(architectures)
    
    def _get_surrogate_predictions(self, architectures):
        """
        Get performance predictions using only the surrogate model.
        
        Args:
            architectures: List of architectures to predict
            
        Returns:
            list: Predicted performance values
        """
        # Check if surrogate model is trained
        if not self.surrogate_model.is_trained:
            logger.warning("Surrogate model not trained, returning random predictions")
            # Return random predictions if not trained
            import random
            if self.higher_is_better:
                return [random.uniform(0.0, 1.0) for _ in range(len(architectures))]
            else:
                return [random.uniform(0.0, 5.0) for _ in range(len(architectures))]
        
        # Make predictions in batches
        predictions = []
        
        for i in range(0, len(architectures), self.predictor_batch_size):
            batch = architectures[i:i + self.predictor_batch_size]
            
            batch_predictions = []
            for arch in batch:
                try:
                    pred = self.surrogate_model.predict_performance(arch)
                    batch_predictions.append(pred)
                except Exception as e:
                    logger.error(f"Error predicting performance: {e}")
                    # Assign neutral prediction
                    if self.higher_is_better:
                        batch_predictions.append(0.5)  # Middle value for accuracy
                    else:
                        batch_predictions.append(1.0)  # Middle value for loss
            
            predictions.extend(batch_predictions)
        
        return predictions
        
    def _get_shared_weights_predictions(self, architectures):
        """
        Get performance predictions using shared weights (ENAS-like approach).
        
        Args:
            architectures: List of architectures to predict
            
        Returns:
            list: Predicted performance values using shared weights
        """
        predictions = []
        
        for arch in architectures:
            try:
                # Check if architecture is similar to anything in the shared weights pool
                # and get an estimate without full training
                arch_str = json.dumps(arch, sort_keys=True)
                similar_performance = None
                
                # Try to find in cache first
                if arch_str in self.evaluated_architectures:
                    performance = self._get_fitness_from_evaluation(self.evaluated_architectures[arch_str])
                    predictions.append(performance)
                    continue
                
                # Try to estimate from shared weights pool
                try:
                    if hasattr(self.evaluator, 'estimate_performance_from_shared_weights'):
                        similar_performance = self.evaluator.estimate_performance_from_shared_weights(
                            arch, self.dataset_name, self.metric
                        )
                except Exception as inner_e:
                    logger.error(f"Error estimating from shared weights: {inner_e}")
                    similar_performance = None
                
                if similar_performance is not None:
                    predictions.append(similar_performance)
                else:
                    # If no shared weights match, use a default value instead of None
                    # This ensures consistent tensor types when combining predictions
                    if self.higher_is_better:
                        default_value = 0.0  # Default low accuracy
                    else:
                        default_value = 1.0  # Default reasonable loss
                    predictions.append(default_value)
                
            except Exception as e:
                logger.error(f"Error getting shared weights prediction: {e}")
                # Use default instead of None to ensure consistent tensor types
                if self.higher_is_better:
                    predictions.append(0.0)  # Default low accuracy
                else:
                    predictions.append(1.0)  # Default reasonable loss
        
        return predictions
    
    def _get_best_architectures(self, architectures, performances, top_k):
        """
        Get the top-k architectures based on performance.
        
        Args:
            architectures: List of architectures
            performances: List of performance values
            top_k: Number of architectures to return
            
        Returns:
            list: Top-k architectures
            list: Performance values of top-k architectures
        """
        # Create (performance, index) pairs
        performance_idx_pairs = [(perf, i) for i, perf in enumerate(performances)]
        
        # Sort based on performance (higher or lower is better)
        if self.higher_is_better:
            performance_idx_pairs.sort(reverse=True)  # Higher is better
        else:
            performance_idx_pairs.sort()  # Lower is better
        
        # Get the indices of the top-k architectures
        top_indices = [idx for _, idx in performance_idx_pairs[:top_k]]
        
        # Extract the top-k architectures and their performances
        top_architectures = [architectures[i] for i in top_indices]
        top_performances = [performances[i] for i in top_indices]
        
        return top_architectures, top_performances
    
    def generate_initial_architectures(self, num_architectures=100):
        """
        Generate initial simple architectures for the first complexity level.
        
        Args:
            num_architectures: Number of initial architectures to generate
            
        Returns:
            list: Initial architectures
        """
        logger.info(f"Generating {num_architectures} initial architectures")
        
        architectures = []
        while len(architectures) < num_architectures:
            # Generate a simple architecture
            architecture = self._sample_architecture_with_complexity(complexity_level=1)
            
            # Validate the architecture
            if self.architecture_space.validate_architecture(architecture):
                architectures.append(architecture)
        
        return architectures
    
    def get_checkpoint_state(self, complexity_level):
        """
        Get the current state for checkpointing.
        
        Args:
            complexity_level: Current complexity level
            
        Returns:
            dict: Current search state
        """
        checkpoint_state = {
            'beam': self.beam,
            'evaluated_architectures': self.evaluated_architectures,
            'higher_is_better': self.higher_is_better,
            'best_fitness': self.best_fitness,
            'best_architecture': self.best_architecture,
            'architecture_space_state': self.architecture_space.__dict__,
            'search_params': {
                'beam_size': self.beam_size,
                'num_expansions': self.num_expansions,
                'max_complexity': self.max_complexity,
                'metric': self.metric,
                'use_shared_weights': self.use_shared_weights,
                'shared_weights_importance': self.shared_weights_importance
            },
            'complexity_level': complexity_level
        }
        
        return checkpoint_state
    
    def restore_from_checkpoint(self, checkpoint):
        """
        Restore search state from a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            
        Returns:
            int: Complexity level to resume from
        """
        try:
            # Restore search state
            search_state = checkpoint['search_state']
            
            # Restore beam and evaluated architectures
            self.beam = search_state['beam']
            self.evaluated_architectures = search_state.get('evaluated_architectures', {})
            
            # Restore best architecture and fitness
            self.best_architecture = checkpoint['best_architecture']
            self.best_fitness = checkpoint['best_fitness']
            
            # Restore history
            self.history = checkpoint['history']
            
            # Get the complexity level to resume from
            complexity_level = search_state['complexity_level']
            
            logger.info(f"Restored search state from checkpoint (complexity level {complexity_level})")
            return complexity_level
            
        except KeyError as e:
            raise SNASException(f"Invalid checkpoint format: missing key {e}")
    
    def search(self, resume_from=None, progress_callback=None, stop_condition=None):
        """
        Run the PNAS search process.
        
        Args:
            resume_from: Path to checkpoint file to resume from
            progress_callback: Optional callback function to report progress (0.0-1.0)
            stop_condition: Optional function that returns True if search should be stopped
            
        Returns:
            dict: Best architecture found
            float: Best fitness value
            dict: Search history
        """
        logger.info("Starting PNAS search")
        
        # Handle resuming from checkpoint
        start_complexity = 1
        if resume_from:
            try:
                checkpoint = self.state_manager.load_checkpoint(resume_from)
                start_complexity = self.restore_from_checkpoint(checkpoint)
                # Start from the next complexity level
                start_complexity += 1
                logger.info(f"Resuming search from complexity level {start_complexity}")
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                logger.info("Starting new search instead")
                start_complexity = 1
                
        # Report initial progress
        if progress_callback:
            progress_callback(0.0, f"Initializing PNAS search (complexity level {start_complexity}/{self.max_complexity})")
        
        # Initialize beam if not resuming or if beam is empty
        if start_complexity == 1 or not self.beam:
            # Generate initial architectures
            initial_architectures = self.generate_initial_architectures(num_architectures=100)
            
            # Evaluate initial architectures
            logger.info("Evaluating initial architectures")
            if progress_callback:
                progress_callback(0.1, f"Complexity level 1/{self.max_complexity}", 
                                  0.0, f"Starting evaluation of {len(initial_architectures)} initial architectures")
            
            # Use the progress-tracking version for initial architectures too
            initial_performances = self._evaluate_architectures_with_progress(
                initial_architectures, 
                fast_mode=True,
                progress_callback=progress_callback,
                current_progress_value=0.1,
                complexity_level=1
            )
            
            # Select top architectures for the beam
            self.beam, beam_performances = self._get_best_architectures(
                initial_architectures, initial_performances, self.beam_size
            )
            
            # Save beam to history
            self.history['complexity_levels'].append(1)
            self.history['beam_architectures'].append(self.beam.copy())
            self.history['beam_performances'].append(beam_performances)
            
            # Train surrogate model on all evaluated architectures
            architectures = list(self.evaluated_architectures.keys())
            arch_objects = [json.loads(arch_str) for arch_str in architectures]
            performances = [self._get_fitness_from_evaluation(self.evaluated_architectures[arch_str]) 
                           for arch_str in architectures]
            
            logger.info(f"Training surrogate model on {len(arch_objects)} architectures")
            if progress_callback:
                progress_callback(0.3, f"Complexity level 1/{self.max_complexity}", 
                                  0.0, f"Starting surrogate model training with {len(arch_objects)} architectures")
                
            # Define a progress reporter for surrogate model training
            def report_surrogate_progress(epoch, total_epochs, loss):
                if progress_callback:
                    sub_progress = epoch / total_epochs if total_epochs > 0 else 0
                    progress_callback(
                        0.3,
                        f"Complexity level 1/{self.max_complexity}",
                        sub_progress,
                        f"Training surrogate model: epoch {epoch}/{total_epochs}, loss: {loss:.6f}"
                    )
            
            # Train surrogate model with progress reporting
            self.surrogate_model.train_surrogate(
                arch_objects, performances, num_epochs=100, batch_size=32,
                progress_callback=report_surrogate_progress
            )
            
            # Save surrogate model training history
            self.history['surrogate_train_losses'].append(self.surrogate_model.train_losses)
            self.history['surrogate_val_losses'].append(self.surrogate_model.val_losses)
            
            # Save best architecture info
            self.history['best_architecture'].append(self.best_architecture)
            self.history['best_fitness'].append(self.best_fitness)
            
            # Save checkpoint if enabled
            if self.checkpoint_frequency > 0 and 1 % self.checkpoint_frequency == 0:
                try:
                    # Get checkpoint state
                    checkpoint_state = self.get_checkpoint_state(complexity_level=1)
                    
                    # Save checkpoint
                    self.state_manager.save_checkpoint(
                        dataset_name=self.dataset_name,
                        search_state=checkpoint_state,
                        best_architecture=self.best_architecture,
                        best_fitness=self.best_fitness,
                        history=self.history,
                        generation=1  # Use complexity level as generation
                    )
                    logger.info(f"Checkpoint saved at complexity level 1")
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
        
        # Progressive complexity search
        for complexity in range(start_complexity, self.max_complexity + 1):
            # Check if search should be stopped
            if stop_condition and stop_condition():
                # Log that we're stopping due to the stop condition
                logger.info(f"Search STOPPING at complexity level {complexity}/{self.max_complexity} due to stop condition")
                if progress_callback:
                    progress_callback(
                        (complexity - start_complexity) / (self.max_complexity - start_complexity + 1),
                        f"Search stopping at complexity level {complexity}/{self.max_complexity}",
                        1.0,  # Set sub-progress to complete
                        f"Search process interrupted by user"
                    )
                    # Sleep to ensure UI updates
                    time.sleep(0.5)
                break
                
            logger.info(f"Complexity level {complexity}/{self.max_complexity}")
            
            # Report progress to UI
            if progress_callback:
                # Calculate overall progress based on complexity level
                progress_value = (complexity - start_complexity) / (self.max_complexity - start_complexity + 1)
                progress_callback(progress_value, f"Complexity level {complexity}/{self.max_complexity}")
            
            # Expand architectures in the beam
            expanded_architectures = []
            total_architectures_to_expand = len(self.beam)
            for i, arch in enumerate(self.beam):
                # Update sub-progress for architecture expansion
                if progress_callback:
                    sub_progress = i / total_architectures_to_expand if total_architectures_to_expand > 0 else 0
                    progress_callback(
                        progress_value,
                        f"Complexity level {complexity}/{self.max_complexity}",
                        sub_progress,
                        f"Expanding architecture {i+1}/{total_architectures_to_expand}"
                    )
                
                # Generate variations of this architecture
                variations = self._expand_architecture(arch)
                expanded_architectures.extend(variations)
            
            logger.info(f"Generated {len(expanded_architectures)} expanded architectures")
            if progress_callback:
                progress_callback(
                    progress_value, 
                    f"Complexity level {complexity}/{self.max_complexity}",
                    1.0,  # Mark sub-progress as complete for this phase
                    f"Generated {len(expanded_architectures)} architectures"
                )
            
            # Predict performance using surrogate model
            predicted_performances = self._predict_performance(expanded_architectures)
            
            # Select top predicted architectures
            top_predicted_architectures, _ = self._get_best_architectures(
                expanded_architectures, predicted_performances, self.beam_size
            )
            
            # Evaluate the top predicted architectures
            logger.info("Evaluating top predicted architectures")
            if progress_callback:
                progress_callback(
                    progress_value, 
                    f"Complexity level {complexity}/{self.max_complexity}",
                    0.0,  # Reset sub-progress for this phase
                    f"Starting evaluation of {len(top_predicted_architectures)} architectures"
                )
                
            # Use the updated evaluation method that reports progress
            actual_performances = self._evaluate_architectures_with_progress(
                top_predicted_architectures, 
                fast_mode=False, 
                progress_callback=progress_callback, 
                current_progress_value=progress_value, 
                complexity_level=complexity
            )
            
            # Update beam with top actual performers
            self.beam, beam_performances = self._get_best_architectures(
                top_predicted_architectures, actual_performances, self.beam_size
            )
            
            # Save to history
            self.history['complexity_levels'].append(complexity)
            self.history['beam_architectures'].append(self.beam.copy())
            self.history['beam_performances'].append(beam_performances)
            
            # Update surrogate model with new data
            architectures = list(self.evaluated_architectures.keys())
            arch_objects = [json.loads(arch_str) for arch_str in architectures]
            performances = [self._get_fitness_from_evaluation(self.evaluated_architectures[arch_str]) 
                           for arch_str in architectures]
            
            logger.info(f"Retraining surrogate model on {len(arch_objects)} architectures")
            if progress_callback:
                progress_callback(
                    progress_value, 
                    f"Complexity level {complexity}/{self.max_complexity}",
                    0.0,  # Reset sub-progress for surrogate training
                    f"Starting surrogate model training with {len(arch_objects)} architectures"
                )
                
            # Define a progress reporter for surrogate model training
            def report_surrogate_progress(epoch, total_epochs, loss):
                if progress_callback:
                    sub_progress = epoch / total_epochs if total_epochs > 0 else 0
                    progress_callback(
                        progress_value,
                        f"Complexity level {complexity}/{self.max_complexity}",
                        sub_progress,
                        f"Training surrogate model: epoch {epoch}/{total_epochs}, loss: {loss:.6f}"
                    )
            
            # Train surrogate model with progress reporting
            self.surrogate_model.train_surrogate(
                arch_objects, performances, num_epochs=100, batch_size=32,
                progress_callback=report_surrogate_progress
            )
            
            # Save surrogate model training history
            self.history['surrogate_train_losses'].append(self.surrogate_model.train_losses)
            self.history['surrogate_val_losses'].append(self.surrogate_model.val_losses)
            
            # Save best architecture info
            self.history['best_architecture'].append(self.best_architecture)
            self.history['best_fitness'].append(self.best_fitness)
            
            # Log best architecture so far
            if self.higher_is_better:
                best_in_beam = max(beam_performances)
            else:
                best_in_beam = min(beam_performances)
                
            logger.info(f"Complexity {complexity} completed. Best in beam: {best_in_beam:.4f}, " 
                      f"Overall best: {self.best_fitness:.4f}")
                      
            # Report completion of this complexity level
            if progress_callback:
                progress_callback(
                    (complexity - start_complexity + 1) / (self.max_complexity - start_complexity + 1),
                    f"Completed complexity level {complexity}/{self.max_complexity}. Best: {self.best_fitness:.4f}"
                )
            
            # Save checkpoint if enabled
            if self.checkpoint_frequency > 0 and complexity % self.checkpoint_frequency == 0:
                try:
                    # Get checkpoint state
                    checkpoint_state = self.get_checkpoint_state(complexity_level=complexity)
                    
                    # Save checkpoint
                    self.state_manager.save_checkpoint(
                        dataset_name=self.dataset_name,
                        search_state=checkpoint_state,
                        best_architecture=self.best_architecture,
                        best_fitness=self.best_fitness,
                        history=self.history,
                        generation=complexity  # Use complexity level as generation
                    )
                    logger.info(f"Checkpoint saved at complexity level {complexity}")
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
        
        # Save final model
        if self.surrogate_model.is_trained:
            try:
                model_path = f"{self.state_manager.results_dir}/{self.dataset_name}_surrogate_model.pt"
                self.surrogate_model.save_model(model_path)
                logger.info(f"Surrogate model saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving surrogate model: {e}")
        
        logger.info("PNAS search completed")
        return self.best_architecture, self.best_fitness, self.history