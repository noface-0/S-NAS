import random
import time
import json
import logging
import numpy as np
from typing import List, Dict, Callable, Any, Optional, Tuple
from collections import defaultdict

from ..utils.exceptions import SNASException, EvaluationError, ArchitectureError
from ..utils.state_manager import SearchStateManager

logger = logging.getLogger(__name__)

class EvolutionarySearch:
    """Evolutionary algorithm for neural architecture search."""
    
    def __init__(self, architecture_space, evaluator, dataset_name, 
                 population_size=20, mutation_rate=0.2, crossover_rate=0.5,
                 generations=10, elite_size=2, tournament_size=3,
                 metric='val_acc', save_history=True, 
                 checkpoint_frequency=0, output_dir="output", results_dir="output/results",
                 enable_progressive=False):
        """
        Initialize the evolutionary search.
        
        Args:
            architecture_space: Space of possible architectures
            evaluator: Component to evaluate architectures
            dataset_name: Name of the dataset to use
            population_size: Size of the population
            mutation_rate: Probability of mutation for each parameter
            crossover_rate: Probability of crossover vs. mutation
            generations: Number of generations to evolve
            elite_size: Number of top individuals to preserve unchanged
            tournament_size: Size of tournaments for parent selection
            metric: Metric to optimize ('val_acc', 'val_loss', 'test_acc', etc.)
            save_history: Whether to save the full history
            checkpoint_frequency: Save checkpoint every N generations (0 to disable)
            output_dir: Directory to store output files
            results_dir: Directory to store result files
            enable_progressive: Whether to enable progressive search (Liu et al., 2018)
        """
        self.architecture_space = architecture_space
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.metric = metric
        self.save_history = save_history
        self.checkpoint_frequency = checkpoint_frequency
        self.enable_progressive = enable_progressive
        
        # Initialize empty population and history
        self.population = []
        self.fitness_scores = []
        
        # Set up state manager for checkpointing
        self.state_manager = SearchStateManager(output_dir, results_dir)
        
        # Set up fitness comparison based on metric type
        # For accuracy metrics, higher is better
        # For loss metrics, lower is better
        self.higher_is_better = not metric.endswith('loss')
        
        # Initialize best values based on whether higher or lower is better
        if self.higher_is_better:
            self.best_fitness = float('-inf')  # Start with worst possible for maximization
        else:
            self.best_fitness = float('inf')   # Start with worst possible for minimization
            
        self.best_architecture = None
        
        # For tracking progress
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_architecture': [],
            'population_diversity': [],
            'evaluation_times': [],
            'complexity_level': [],  # For progressive search - tracks complexity level by generation
            'metric': metric,
            'metric_type': 'loss' if metric.endswith('loss') else 'accuracy'
        }
        
        # For tracking all evaluated architectures to avoid duplicates
        self.evaluated_architectures = {}
        
        # For progressive search (Liu et al., 2018)
        if self.enable_progressive:
            self.complexity_level = 1  # Start with simplest architectures
            self.max_complexity_level = 3  # Maximum complexity level
            self.complexity_transition_point = generations // 3  # When to increase complexity
            logger.info(f"Progressive search enabled: starting with complexity level 1")
        
        logger.info(f"Evolutionary search initialized with metric: {metric} "
                   f"({'higher is better' if self.higher_is_better else 'lower is better'})")
    
    def initialize_population(self):
        """Create initial random population of architectures."""
        logger.info("Initializing population with %d individuals", self.population_size)
        
        self.population = []
        while len(self.population) < self.population_size:
            # Generate a random architecture (possibly constrained by complexity level)
            if self.enable_progressive:
                architecture = self._sample_architecture_with_complexity(self.complexity_level)
            else:
                architecture = self.architecture_space.sample_random_architecture()
                
            # Fix parameters that need to be lists for non-progressive search
            if 'use_skip_connections' in architecture and not isinstance(architecture['use_skip_connections'], list):
                architecture['use_skip_connections'] = [architecture['use_skip_connections']] * architecture['num_layers']
            
            # Validate the architecture
            if self.architecture_space.validate_architecture(architecture):
                # Add to population
                self.population.append(architecture)
        
        logger.info("Finished initializing population")
        
    def _sample_architecture_with_complexity(self, complexity_level):
        """
        Sample an architecture with the specified complexity level.
        
        This implements the progressive search approach from Liu et al. (2018),
        where architectures start simple and grow more complex over generations.
        
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
            architecture['num_layers'] = min(4, architecture['num_layers'])
            
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
                
            elif architecture['network_type'] == 'mlp':  # Use elif instead of else
                # Initialize hidden_units if it doesn't exist
                if 'hidden_units' not in architecture:
                    architecture['hidden_units'] = [512 for _ in range(architecture['num_layers'])]
                else:
                    architecture['hidden_units'] = [min(h, 512) for h in architecture['hidden_units']]
            
            # Simplify activation functions - just use ReLU at level 1
            architecture['activations'] = ['relu' for _ in range(architecture['num_layers'])]
                
        elif complexity_level == 2:
            # Medium complexity - moderate layers, some advanced features
            architecture['num_layers'] = min(8, architecture['num_layers'])
            
            # Allow more network types at level 2
            medium_types = ['cnn', 'mlp', 'enhanced_mlp', 'resnet', 'mobilenet']
            if architecture['network_type'] not in medium_types:
                if random.random() < 0.5:  # 50% chance to keep original type
                    architecture['network_type'] = random.choice(medium_types)
            
            # Some skip connections and batch norm allowed
            architecture['use_batch_norm'] = random.choice([True, False])
            
        # Level 3 (maximum complexity) uses the full architecture space as defined
        
        # Handle special parameter types
        if 'use_skip_connections' in architecture and not isinstance(architecture['use_skip_connections'], list):
            architecture['use_skip_connections'] = [architecture['use_skip_connections']] * architecture['num_layers']
            
        # Ensure architecture is consistent after modifications
        architecture = self._ensure_architecture_consistency(architecture)
        
        return architecture
        
    def _ensure_architecture_consistency(self, architecture):
        """
        Ensure an architecture's parameters are consistent after modification.
        
        Args:
            architecture: Architecture to make consistent
            
        Returns:
            dict: Consistent architecture
        """
        num_layers = architecture['num_layers']
        network_type = architecture['network_type']
        
        # Make sure layer-specific parameters have the right length
        if network_type in ['cnn', 'resnet', 'mobilenet']:
            for param in ['filters', 'kernel_sizes', 'activations', 'use_skip_connections']:
                if param in architecture:
                    # Truncate or extend the list to match num_layers
                    if len(architecture[param]) > num_layers:
                        architecture[param] = architecture[param][:num_layers]
                    elif len(architecture[param]) < num_layers:
                        # Extend with random values from the original list
                        while len(architecture[param]) < num_layers:
                            architecture[param].append(
                                random.choice(architecture[param])
                            )
        
        elif network_type in ['mlp', 'enhanced_mlp']:
            for param in ['hidden_units', 'activations']:
                if param in architecture:
                    # Truncate or extend the list to match num_layers
                    if len(architecture[param]) > num_layers:
                        architecture[param] = architecture[param][:num_layers]
                    elif len(architecture[param]) < num_layers:
                        # Extend with random values from the original list
                        while len(architecture[param]) < num_layers:
                            architecture[param].append(
                                random.choice(architecture[param])
                            )
                            
        return architecture
    
    def evaluate_population(self, fast_mode=False):
        """
        Evaluate the fitness of each architecture in the population.
        
        Args:
            fast_mode: If True, use a reduced evaluation protocol for faster iterations
            
        Returns:
            list: Fitness scores for each architecture
        """
        logger.info("Evaluating population fitness")
        
        fitness_scores = []
        start_time = time.time()
        
        for i, architecture in enumerate(self.population):
            # Check if we've evaluated this architecture before
            arch_str = json.dumps(architecture, sort_keys=True)
            if arch_str in self.evaluated_architectures:
                # Use cached result
                evaluation = self.evaluated_architectures[arch_str]
                logger.info(f"Architecture {i+1}/{len(self.population)} already evaluated, using cached result")
            else:
                # Evaluate the architecture
                logger.info(f"Evaluating architecture {i+1}/{len(self.population)}")
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
                        fitness_scores.append(float('-inf'))
                    else:
                        fitness_scores.append(float('inf'))
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error evaluating architecture: {e}")
                    # Create a more detailed error report
                    error_msg = f"Failed to evaluate architecture: {type(e).__name__}: {str(e)}"
                    error = EvaluationError(error_msg, architecture=architecture, details=str(e))
                    # Assign a very poor fitness
                    if self.higher_is_better:
                        fitness_scores.append(float('-inf'))
                    else:
                        fitness_scores.append(float('inf'))
                    continue
            
            # Extract fitness from evaluation result using helper method
            fitness = self._get_fitness_from_evaluation(evaluation)
            
            # Add debugging information
            logger.debug(f"Architecture {i+1} fitness: {fitness:.4f} "
                        f"({'better' if self.is_better_fitness(fitness, self.best_fitness) else 'worse'} "
                        f"than current best: {self.best_fitness:.4f})")
            
            fitness_scores.append(fitness)
            
            # Update best architecture if better
            if self.is_better_fitness(fitness, self.best_fitness):
                self.best_fitness = fitness
                self.best_architecture = architecture.copy()
                logger.info(f"New best architecture found! Fitness: {fitness:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        
        self.fitness_scores = fitness_scores
        return fitness_scores
    
    def is_better_fitness(self, candidate, reference):
        """
        Determine if a candidate fitness is better than a reference fitness.
        
        Args:
            candidate: Candidate fitness value
            reference: Reference fitness value to compare against
            
        Returns:
            bool: True if candidate is better than reference, False otherwise
        """
        if self.higher_is_better:
            return candidate > reference
        else:
            return candidate < reference
    
    def _select_parents(self, population, fitness_scores):
        """
        Select parents using tournament selection.
        
        Args:
            population: List of architectures
            fitness_scores: Fitness score for each architecture
            
        Returns:
            list: Selected parent architectures
        """
        selected_parents = []
        
        # Select population_size parents
        for _ in range(self.population_size):
            # Randomly select tournament_size candidates
            candidate_indices = random.sample(range(len(population)), self.tournament_size)
            candidate_fitness = [fitness_scores[i] for i in candidate_indices]
            
            # Select the best candidate from the tournament
            if self.higher_is_better:
                best_idx = candidate_fitness.index(max(candidate_fitness))
            else:
                best_idx = candidate_fitness.index(min(candidate_fitness))
                
            winner_idx = candidate_indices[best_idx]
            selected_parents.append(population[winner_idx])
            
        return selected_parents
    
    def _crossover(self, parent1, parent2):
        """
        Combine two parent architectures to create a child.
        
        Args:
            parent1: First parent architecture
            parent2: Second parent architecture
            
        Returns:
            dict: Child architecture
        """
        # First, check if network types are the same
        if parent1.get('network_type', '') != parent2.get('network_type', ''):
            # Network types are different - use one parent as the base
            # and just apply mutation to it rather than trying to combine incompatible architectures
            return parent1.copy() if random.random() < 0.5 else parent2.copy()
        
        # Start with an empty child architecture
        child = {}
        
        # For each parameter in the architecture
        for key in set(list(parent1.keys()) + list(parent2.keys())):
            # Skip input shape and num_classes, they should stay the same as parent1
            if key in ['input_shape', 'num_classes']:
                child[key] = parent1[key]
                continue
                
            # If a key is in only one parent, take it from that parent
            if key not in parent1:
                child[key] = parent2[key]
                continue
            if key not in parent2:
                child[key] = parent1[key]
                continue
                
            # Both parents have this key, perform crossover
            if isinstance(parent1[key], list) and isinstance(parent2[key], list) and len(parent1[key]) > 1 and len(parent2[key]) > 1:
                # For list parameters (like filters per layer), perform crossover
                # at a random point in the list
                crossover_point = random.randint(1, min(len(parent1[key]), len(parent2[key]))-1)
                child[key] = parent1[key][:crossover_point] + parent2[key][crossover_point:]
            else:
                # For scalar parameters, randomly choose from either parent
                child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        
        # Make sure num_layers matches the length of layer lists
        if 'num_layers' in child:
            network_type = child.get('network_type', 'cnn')
            list_params = []
            
            if network_type == 'mlp':
                list_params = ['hidden_units', 'activations']
            else:  # cnn, resnet, mobilenet
                list_params = ['filters', 'kernel_sizes', 'activations', 'use_skip_connections']
                
            for param in list_params:
                if param in child:
                    child['num_layers'] = len(child[param])
                    break
                
        return child
    
    def create_next_generation(self):
        """
        Create the next generation through elitism, crossover, and mutation.
        
        Returns:
            list: New population
        """
        # Sort population by fitness (better fitness first)
        if self.higher_is_better:
            sorted_indices = sorted(range(len(self.fitness_scores)), 
                                   key=lambda i: self.fitness_scores[i], reverse=True)
        else:
            sorted_indices = sorted(range(len(self.fitness_scores)), 
                                   key=lambda i: self.fitness_scores[i], reverse=False)
        
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [self.fitness_scores[i] for i in sorted_indices]
        
        new_population = []
        
        # Elitism: keep the best individuals unchanged
        for i in range(min(self.elite_size, len(sorted_population))):
            new_population.append(sorted_population[i])
        
        # Fill the rest of the population with offspring
        parents = self._select_parents(sorted_population, sorted_fitness)
        
        while len(new_population) < self.population_size:
            # Decide whether to use crossover or just mutation
            if random.random() < self.crossover_rate:
                # Select two parents and create offspring via crossover
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
            else:
                # Just select one parent
                parent = random.choice(parents)
                child = parent.copy()
            
            # Apply random mutations
            child = self.architecture_space.mutate_architecture(child, self.mutation_rate)
            
            # For progressive search, ensure the child matches the current complexity level
            if self.enable_progressive:
                child = self._adapt_architecture_to_complexity(child, self.complexity_level)
            
            # Fix use_skip_connections if it's a boolean instead of a list (for any search mode)
            if 'use_skip_connections' in child and not isinstance(child['use_skip_connections'], list):
                child['use_skip_connections'] = [child['use_skip_connections']] * child['num_layers']
            
            # Validate the child
            if self.architecture_space.validate_architecture(child):
                new_population.append(child)
        
        return new_population
        
    def _adapt_architecture_to_complexity(self, architecture, complexity_level):
        """
        Adapt an architecture to match the current complexity level.
        
        This ensures that architectures stay within the appropriate complexity
        level during the progressive search process.
        
        Args:
            architecture: Architecture to adapt
            complexity_level: Current complexity level
            
        Returns:
            dict: Adapted architecture
        """
        if complexity_level == 1:
            # Enforce simplest constraints at level 1
            architecture['num_layers'] = min(3, architecture['num_layers'])
            
            # Keep simpler network types
            simple_types = ['cnn', 'mlp']
            if architecture['network_type'] not in simple_types:
                # Don't change randomly - find closest simple type
                if 'mlp' in architecture['network_type']:
                    architecture['network_type'] = 'mlp'
                else:
                    architecture['network_type'] = 'cnn'
                    
            # Limit other parameters based on network type
            if architecture['network_type'] == 'cnn':
                architecture['filters'] = [min(f, 64) for f in architecture['filters']]
            elif architecture['network_type'] == 'mlp' and 'hidden_units' in architecture:
                architecture['hidden_units'] = [min(h, 256) for h in architecture['hidden_units']]
        
        elif complexity_level == 2:
            # Medium complexity level
            architecture['num_layers'] = min(5, architecture['num_layers'])
            
            # Allow only certain network types at level 2
            medium_types = ['cnn', 'mlp', 'enhanced_mlp', 'resnet']
            if architecture['network_type'] not in medium_types:
                # Try to map to a similar medium-complexity type
                if 'mlp' in architecture['network_type']:
                    architecture['network_type'] = 'enhanced_mlp'
                elif 'net' in architecture['network_type']:
                    architecture['network_type'] = 'resnet'
                else:
                    architecture['network_type'] = 'cnn'
        
        # No additional constraints for level 3 (full complexity)
        
        # Ensure architecture consistency
        architecture = self._ensure_architecture_consistency(architecture)
        
        return architecture
    
    def calculate_diversity(self):
        """
        Calculate population diversity based on parameter distributions.
        
        Returns:
            float: Diversity score (higher is more diverse)
        """
        # Count occurrences of each parameter value
        value_counts = defaultdict(lambda: defaultdict(int))
        
        for arch in self.population:
            for param, value in arch.items():
                if isinstance(value, list):
                    # For list parameters, count each element separately
                    for i, v in enumerate(value):
                        value_counts[f"{param}_{i}"][str(v)] += 1
                else:
                    # For scalar parameters
                    value_counts[param][str(value)] += 1
        
        # Calculate entropy for each parameter
        entropies = []
        for param, counts in value_counts.items():
            total = sum(counts.values())
            entropy = 0
            for count in counts.values():
                p = count / total
                entropy -= p * (p > 0 and np.log2(p) or 0)  # Shannon entropy
            entropies.append(entropy)
        
        # Average entropy across all parameters
        if entropies:
            return sum(entropies) / len(entropies)
        return 0.0
    
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
        # If metric not found, use a sensible default
        if self.metric == 'val_acc':
            fitness = evaluation.get('best_val_acc', 0.0)
        elif self.metric == 'val_loss':
            # Note: Evaluator tracks best_val_loss as the lowest seen validation loss
            fitness = evaluation.get('best_val_loss', float('inf'))
            # If best_val_loss isn't available, try using the last validation loss
            if fitness == float('inf') and 'val_losses' in evaluation and evaluation['val_losses']:
                fitness = evaluation['val_losses'][-1]
        elif self.metric == 'test_acc':
            fitness = evaluation.get('test_acc', 0.0)
        elif self.metric == 'test_loss':
            fitness = evaluation.get('test_loss', float('inf'))
        else:
            # For any other metric, try to get it directly
            fitness = evaluation.get(self.metric, 0.0 if self.higher_is_better else float('inf'))
        
        return fitness
    
    def get_checkpoint_state(self):
        """
        Get the current state for checkpointing.
        
        Returns:
            dict: Current search state
        """
        checkpoint_state = {
            'population': self.population,
            'fitness_scores': self.fitness_scores,
            'evaluated_architectures': self.evaluated_architectures,
            'higher_is_better': self.higher_is_better,
            'best_fitness': self.best_fitness,
            'best_architecture': self.best_architecture,
            'architecture_space_state': self.architecture_space.__dict__,
            'search_params': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'generations': self.generations,
                'elite_size': self.elite_size,
                'tournament_size': self.tournament_size,
                'metric': self.metric,
                'enable_progressive': self.enable_progressive
            }
        }
        
        # Add progressive search state if enabled
        if self.enable_progressive:
            checkpoint_state['complexity_level'] = self.complexity_level
            checkpoint_state['max_complexity_level'] = self.max_complexity_level
            checkpoint_state['complexity_transition_point'] = self.complexity_transition_point
        
        return checkpoint_state
    
    def restore_from_checkpoint(self, checkpoint):
        """
        Restore search state from a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            
        Returns:
            int: Generation to resume from
        """
        try:
            # Restore search state
            search_state = checkpoint['search_state']
            
            # Restore population and fitness
            self.population = search_state['population']
            self.fitness_scores = search_state.get('fitness_scores', [])
            
            # Restore evaluated architectures
            self.evaluated_architectures = search_state.get('evaluated_architectures', {})
            
            # Restore best architecture and fitness
            self.best_architecture = checkpoint['best_architecture']
            self.best_fitness = checkpoint['best_fitness']
            
            # Restore progressive search state if present and enabled
            if self.enable_progressive:
                if 'complexity_level' in search_state:
                    self.complexity_level = search_state['complexity_level']
                    logger.info(f"Restored complexity level: {self.complexity_level}")
                
                if 'max_complexity_level' in search_state:
                    self.max_complexity_level = search_state['max_complexity_level']
                
                if 'complexity_transition_point' in search_state:
                    self.complexity_transition_point = search_state['complexity_transition_point']
            
            # Restore history
            if self.save_history:
                self.history = checkpoint['history']
                
                # Ensure complexity_level is in history if progressive search is enabled
                if self.enable_progressive and 'complexity_level' not in self.history:
                    self.history['complexity_level'] = []
            
            # Get the generation to resume from
            generation = checkpoint['generation']
            
            logger.info(f"Restored search state from checkpoint (generation {generation})")
            return generation
            
        except KeyError as e:
            raise SNASException(f"Invalid checkpoint format: missing key {e}")
    
    def evolve(self, fast_mode_generations=0, resume_from=None):
        """
        Run the evolutionary search process.
        
        Args:
            fast_mode_generations: Number of initial generations to run in fast mode
                                   for quicker exploration
            resume_from: Path to checkpoint file to resume from
        
        Returns:
            dict: The best architecture found
        """
        logger.info("Starting evolutionary search")
        
        # Handle resuming from checkpoint
        start_generation = 0
        if resume_from:
            try:
                checkpoint = self.state_manager.load_checkpoint(resume_from)
                start_generation = self.restore_from_checkpoint(checkpoint)
                # Start from the next generation
                start_generation += 1
                logger.info(f"Resuming search from generation {start_generation}")
                
                # Restore complexity level for progressive search if resuming
                if self.enable_progressive and 'complexity_level' in checkpoint['search_state']:
                    self.complexity_level = checkpoint['search_state']['complexity_level']
                    logger.info(f"Restored complexity level: {self.complexity_level}")
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                logger.info("Starting new search instead")
                start_generation = 0
        
        # Initialize population if not resuming or if population is empty
        if start_generation == 0 or not self.population:
            self.initialize_population()
        
        for generation in range(start_generation, self.generations):
            logger.info(f"Generation {generation+1}/{self.generations}")
            
            # Use fast mode for the first few generations to explore the space quickly
            use_fast_mode = generation < fast_mode_generations
            if use_fast_mode:
                logger.info("Using fast evaluation mode")
                
            # For progressive search, check if we should increase complexity
            if self.enable_progressive:
                # Update complexity level at transition points
                transition_point = (self.complexity_level * self.generations) // (self.max_complexity_level + 1)
                if generation >= transition_point and self.complexity_level < self.max_complexity_level:
                    self.complexity_level += 1
                    logger.info(f"Increasing architecture complexity to level {self.complexity_level}")
                
                # Add current complexity level to history
                if self.save_history:
                    self.history['complexity_level'].append(self.complexity_level)
            
            try:
                # Evaluate fitness of each architecture
                fitness_scores = self.evaluate_population(fast_mode=use_fast_mode)
                
                # Record statistics for this generation
                if len(fitness_scores) > 0:
                    if self.higher_is_better:
                        best_fitness = max(fitness_scores)
                        best_idx = fitness_scores.index(best_fitness)
                    else:
                        best_fitness = min(fitness_scores)
                        best_idx = fitness_scores.index(best_fitness)
                        
                    avg_fitness = sum(fitness_scores) / len(fitness_scores)
                    best_arch = self.population[best_idx].copy()
                    diversity = self.calculate_diversity()
                    
                    if self.enable_progressive:
                        logger.info(f"Generation stats: Complexity: {self.complexity_level}, "
                                  f"Avg fitness: {avg_fitness:.4f}, Best fitness: {best_fitness:.4f}, "
                                  f"Diversity: {diversity:.4f}")
                    else:
                        logger.info(f"Generation stats: Avg fitness: {avg_fitness:.4f}, "
                                  f"Best fitness: {best_fitness:.4f}, Diversity: {diversity:.4f}")
                    
                    # Save history
                    if self.save_history:
                        self.history['generations'].append(generation)
                        self.history['best_fitness'].append(best_fitness)
                        self.history['avg_fitness'].append(avg_fitness)
                        self.history['best_architecture'].append(best_arch)
                        self.history['population_diversity'].append(diversity)
                else:
                    logger.warning("No valid fitness scores in this generation")
                
                # Save checkpoint if enabled
                if self.checkpoint_frequency > 0 and (generation + 1) % self.checkpoint_frequency == 0:
                    try:
                        # Get checkpoint state
                        checkpoint_state = self.get_checkpoint_state()
                        
                        # Save checkpoint
                        self.state_manager.save_checkpoint(
                            dataset_name=self.dataset_name,
                            search_state=checkpoint_state,
                            best_architecture=self.best_architecture,
                            best_fitness=self.best_fitness,
                            history=self.history,
                            generation=generation
                        )
                        logger.info(f"Checkpoint saved at generation {generation+1}")
                    except Exception as e:
                        logger.error(f"Error saving checkpoint: {e}")
                
                # Create next generation (except for the last iteration)
                if generation < self.generations - 1:
                    self.population = self.create_next_generation()
            
            except Exception as e:
                logger.error(f"Error in generation {generation+1}: {e}")
                # Try to save emergency checkpoint
                try:
                    checkpoint_state = self.get_checkpoint_state()
                    self.state_manager.save_checkpoint(
                        dataset_name=self.dataset_name,
                        search_state=checkpoint_state,
                        best_architecture=self.best_architecture,
                        best_fitness=self.best_fitness,
                        history=self.history,
                        generation=generation,
                        use_timestamp=True
                    )
                    logger.info("Emergency checkpoint saved")
                except Exception as checkpoint_e:
                    logger.error(f"Failed to save emergency checkpoint: {checkpoint_e}")
                
                # Continue with next generation
                continue
        
        # Final evaluation of the best architecture (if we used fast mode)
        if fast_mode_generations > 0 and self.best_architecture:
            logger.info("Performing final evaluation of best architecture")
            
            # Evaluate with full protocol
            try:
                evaluation = self.evaluator.evaluate(
                    self.best_architecture, self.dataset_name, fast_mode=False
                )
                
                # Update best fitness with full evaluation result
                self.best_fitness = self._get_fitness_from_evaluation(evaluation)
                logger.info(f"Final evaluation result: {self.best_fitness:.4f}")
                
                # Save final checkpoint
                try:
                    checkpoint_state = self.get_checkpoint_state()
                    self.state_manager.save_checkpoint(
                        dataset_name=self.dataset_name,
                        search_state=checkpoint_state,
                        best_architecture=self.best_architecture,
                        best_fitness=self.best_fitness,
                        history=self.history,
                        generation=self.generations - 1,
                        use_timestamp=True
                    )
                    logger.info("Final checkpoint saved")
                except Exception as checkpoint_e:
                    logger.error(f"Failed to save final checkpoint: {checkpoint_e}")
                
            except EvaluationError as e:
                logger.error(f"Error in final evaluation: {e}")
                logger.warning("Using previous best fitness value")
            except Exception as e:
                error_msg = f"Unexpected error in final evaluation: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                # Create a more detailed error report
                EvaluationError(error_msg, architecture=self.best_architecture, details=str(e))
        
        logger.info("Evolutionary search completed")
        return self.best_architecture, self.best_fitness, self.history